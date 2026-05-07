"""
train_karman_2d.py — Conditional DDPM training on 2D Kármán-vortex Tucker factors.

Tucker rank = [10, 128, 30]. Conditions:
  - Initial frame (U_ic, Vh_ic) as token-level context
  - Five scalars (niu, cx, cy, r, Re) via ScalarEmbedder + AdaLN

Token layout (1478 total):
  [COND (158) | MAIN (1320)]  — only MAIN tokens are noised/denoised.

Sampling visualisation: for each test sample, pick `n_frames_vis` evenly
spaced time indices and render them as matplotlib heatmaps. Output composite
is 2 * n_vis rows × n_frames_vis cols (real row above gen row for each sample).
"""

import argparse
import glob
import os
import sys
from copy import deepcopy
from time import time

import numpy as np
import torch
import yaml
import wandb
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

_EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '${REPO_ROOT}/video')

from diffusion import create_diffusion
from models.nn import update_ema

from dataset_karman_2d import KarmanTucker2DDataset, reconstruct_video
from model_karman_2d_dit import (
    build_karman_2d_dit,
    FLAT_MAIN, FLAT_COND,
    FLAT_UT, FLAT_UY, FLAT_G,
    R_T, R_Y, T_DIM, H_DIM,
)


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------

def pack(batch):
    """
    Pack dataset batch into (x_flat, cond_flat, niu, cx, cy, r, re).
    """
    U_T   = batch['U_T']    # (B, 200, r_T)
    U_Y   = batch['U_Y']    # (B, 128, r_Y)
    G     = batch['G']      # (B, r_T, 128, r_Y)
    U_ic  = batch['U_ic']   # (B, 128, r_ic)
    Vh_ic = batch['Vh_ic']  # (B, r_ic, 128)
    niu   = batch['niu']
    cx    = batch['cx']
    cy    = batch['cy']
    r     = batch['r']
    re    = batch['Re']

    x_flat    = torch.cat([U_T.flatten(1), U_Y.flatten(1), G.flatten(1)], dim=1)
    cond_flat = torch.cat([U_ic.flatten(1), Vh_ic.flatten(1)],            dim=1)
    return x_flat, cond_flat, niu, cx, cy, r, re


def unpack_x(x_flat, B):
    c0, c1, c2 = x_flat.split([FLAT_UT, FLAT_UY, FLAT_G], dim=1)
    U_T = c0.reshape(B, T_DIM, R_T)
    U_Y = c1.reshape(B, H_DIM, R_Y)
    G   = c2.reshape(B, R_T, H_DIM, R_Y)
    return U_T, U_Y, G


# ---------------------------------------------------------------------------
# 10-frame heatmap grid
# ---------------------------------------------------------------------------

def render_frame_grid(videos_real, videos_gen, n_frames: int,
                       epoch: int, step: int, mean_rel_err: float):
    """
    Render composite grid for wandb.
    Layout: 2*B rows × n_frames cols.
      row 2i   : sample i real, n_frames evenly spaced frames
      row 2i+1 : sample i gen,  same time indices
    Per-sample vlim shared across real/gen rows.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    B = len(videos_real)
    T = videos_real[0].shape[0]
    frame_idx = np.linspace(0, T - 1, n_frames).astype(int)

    fig, axes = plt.subplots(
        2 * B, n_frames,
        figsize=(1.4 * n_frames, 1.4 * 2 * B),
        dpi=90,
    )
    if 2 * B == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(B):
        v_real = videos_real[i]   # (T, H, W)
        v_gen  = videos_gen[i]
        vlim   = float(np.percentile(np.abs(v_real), 98))
        norm   = Normalize(-vlim, vlim)

        for j, t_idx in enumerate(frame_idx):
            ax_r = axes[2 * i,     j]
            ax_g = axes[2 * i + 1, j]
            ax_r.imshow(v_real[t_idx], cmap='RdBu_r', norm=norm)
            ax_g.imshow(v_gen[t_idx],  cmap='RdBu_r', norm=norm)
            for ax in (ax_r, ax_g):
                ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax_r.set_title(f't={t_idx}', fontsize=8)
            if j == 0:
                ax_r.set_ylabel(f'real #{i}', fontsize=8)
                ax_g.set_ylabel(f'gen  #{i}', fontsize=8)

    fig.suptitle(
        f'epoch={epoch}  step={step}  rel_err={mean_rel_err:.4f}',
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


# ---------------------------------------------------------------------------
# Sampling + visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_and_visualize(
    wrapper, diffusion_sample, test_dataset, device,
    train_dataset, n_vis: int, n_frames_vis: int,
    step: int, epoch: int, vis_dir: str,
):
    wrapper.eval()
    B = n_vis

    vis_indices = torch.randperm(len(test_dataset))[:B].tolist()
    samples_list = [test_dataset[i] for i in vis_indices]
    test_batch = {
        k: (torch.stack([s[k] for s in samples_list])
            if k != 'idx'
            else [s[k] for s in samples_list])
        for k in samples_list[0].keys()
    }

    x_flat, cond_flat, niu, cx, cy, r, re = pack(test_batch)
    x_flat    = x_flat[:B].to(device)
    cond_flat = cond_flat[:B].to(device)
    niu = niu[:B].to(device); cx = cx[:B].to(device)
    cy  = cy[:B].to(device);  r  = r[:B].to(device)
    re  = re[:B].to(device)

    noise   = torch.randn(B, FLAT_MAIN, device=device)
    samples = diffusion_sample.p_sample_loop(
        wrapper, noise.shape, noise=noise,
        clip_denoised=False,
        model_kwargs={'cond_flat': cond_flat,
                      'niu': niu, 'cx': cx, 'cy': cy, 'r': r, 're': re},
        device=device, progress=False,
    )

    UT_gen,  UY_gen,  G_gen  = unpack_x(samples.float().cpu(), B)
    UT_real, UY_real, G_real = unpack_x(x_flat.float().cpu(),  B)

    rel_errors, videos_real, videos_gen = [], [], []
    for i in range(B):
        def _recon(UT_i, UY_i, G_i):
            UT_dn = train_dataset.denorm(UT_i, 'UT').cpu().numpy()
            UY_dn = train_dataset.denorm(UY_i, 'UY').cpu().numpy()
            G_dn  = train_dataset.denorm(G_i,  'G').cpu().numpy()
            return reconstruct_video(UT_dn, UY_dn, G_dn).astype(np.float32)

        v_real = _recon(UT_real[i], UY_real[i], G_real[i])
        v_gen  = _recon(UT_gen[i],  UY_gen[i],  G_gen[i])

        rel_err = float(np.linalg.norm(v_real - v_gen) /
                        (np.linalg.norm(v_real) + 1e-8))
        rel_errors.append(rel_err)
        videos_real.append(v_real)
        videos_gen.append(v_gen)

    mean_rel_err = float(np.mean(rel_errors))
    print(f'  rel_err: {[f"{e:.4f}" for e in rel_errors]}  mean={mean_rel_err:.4f}',
          flush=True)

    log_dict = {'val/rel_error_mean': mean_rel_err}
    try:
        import matplotlib.pyplot as plt
        fig = render_frame_grid(videos_real, videos_gen, n_frames_vis,
                                 epoch, step, mean_rel_err)
        log_dict['val/frame_grid'] = wandb.Image(fig)
        plt.close(fig)
    except Exception as e:
        print(f'  [frame-grid render skipped: {e}]', flush=True)

    wrapper.train()
    return log_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train 2D Kármán Tucker DiT')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data_dir',          type=str,   default=None)
    parser.add_argument('--test_data_dir',     type=str,   default=None)
    parser.add_argument('--batch_size',        type=int,   default=None)
    parser.add_argument('--lr',                type=float, default=None)
    parser.add_argument('--n_epochs',          type=int,   default=None)
    parser.add_argument('--n_steps',           type=int,   default=None)
    parser.add_argument('--hidden_size',       type=int,   default=None)
    parser.add_argument('--depth',             type=int,   default=None)
    parser.add_argument('--num_heads',         type=int,   default=None)
    parser.add_argument('--mlp_ratio',         type=float, default=None)
    parser.add_argument('--log_every',         type=int,   default=None)
    parser.add_argument('--vis_every_epochs',  type=int,   default=None)
    parser.add_argument('--save_every_epochs', type=int,   default=None)
    parser.add_argument('--n_test',            type=int,   default=None)
    parser.add_argument('--n_vis',             type=int,   default=None)
    parser.add_argument('--n_frames_vis',      type=int,   default=None)
    parser.add_argument('--T_diffusion',       type=int,   default=None)
    parser.add_argument('--sample_steps',      type=int,   default=None)
    parser.add_argument('--noise_schedule',    type=str,   default=None)
    parser.add_argument('--ema_rate',          type=float, default=None)
    parser.add_argument('--grad_clip',         type=float, default=None)
    parser.add_argument('--mixed_precision',   type=str,   default=None)
    parser.add_argument('--outdir',            type=str,   default=None)
    parser.add_argument('--wandb_project',     type=str,   default=None)
    parser.add_argument('--wandb_run',         type=str,   default=None)
    parser.add_argument('--wandb_entity',      type=str,   default=None)
    parser.add_argument('--device',            type=str,   default=None)
    parser.add_argument('--seed',              type=int,   default=None)
    parser.add_argument('--resume',            type=str,   default=None)
    cli = parser.parse_args()

    defaults = dict(
        data_dir          = os.path.join(_EXP, 'data', 'tucker_karman_rT10_rX128_rY30'),
        test_data_dir     = None,
        batch_size        = 32,
        lr                = 1e-4,
        n_epochs          = None,
        n_steps           = 200000,
        hidden_size       = 512,
        depth             = 8,
        num_heads         = 8,
        mlp_ratio         = 4.0,
        log_every         = 500,
        vis_every_epochs  = 20,
        save_every_epochs = 100,
        n_test            = 20,
        n_vis             = 4,
        n_frames_vis      = 10,
        T_diffusion       = 1000,
        sample_steps      = 250,
        noise_schedule    = 'linear',
        ema_rate          = 0.9999,
        grad_clip         = 1.0,
        mixed_precision   = True,
        outdir            = os.path.join(_EXP, 'output', 'train'),
        wandb_project     = '<PROJECT>',
        wandb_run         = 'karman2d_tucker_dit_v1',
        wandb_entity      = None,
        device            = 'auto',
        seed              = 0,
        resume            = None,
    )

    if cli.config is not None:
        cfg_path = cli.config if os.path.isabs(cli.config) else \
                   os.path.join(_EXP, cli.config)
        with open(cfg_path) as f:
            yaml_cfg = yaml.safe_load(f)
        for key in ('data_dir', 'test_data_dir', 'outdir'):
            if key in yaml_cfg and yaml_cfg[key] and not os.path.isabs(yaml_cfg[key]):
                yaml_cfg[key] = os.path.normpath(os.path.join(_EXP, yaml_cfg[key]))
        defaults.update({k: v for k, v in yaml_cfg.items() if v is not None})

    for key, val in vars(cli).items():
        if key == 'config':
            continue
        if val is not None:
            if key == 'mixed_precision':
                defaults[key] = val.lower() != 'false'
            else:
                defaults[key] = val

    args = argparse.Namespace(**defaults)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}', flush=True)

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ckpt_dir = os.path.join(args.outdir, 'checkpoints')
    vis_dir  = os.path.join(args.outdir, 'vis')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir,  exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────
    if args.test_data_dir is not None:
        train_dataset = KarmanTucker2DDataset(args.data_dir,
                                                split='all', device=device)
        test_dataset  = KarmanTucker2DDataset(args.test_data_dir,
                                                split='all', device=device,
                                                external_stats=train_dataset.stats)
    else:
        full_ds = KarmanTucker2DDataset(args.data_dir, split='all', device=device)
        N_total = len(full_ds)
        test_indices = list(range(N_total - args.n_test, N_total))
        train_dataset = KarmanTucker2DDataset(args.data_dir, test_indices=test_indices,
                                                split='train', device=device)
        test_dataset  = KarmanTucker2DDataset(args.data_dir, test_indices=test_indices,
                                                split='test',  device=device)
    print(f'Train: {len(train_dataset)}  |  Test: {len(test_dataset)}', flush=True)

    n_vis = min(args.n_vis, len(test_dataset))

    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    cfg = {
        'hidden_size': args.hidden_size,
        'depth':       args.depth,
        'num_heads':   args.num_heads,
        'mlp_ratio':   args.mlp_ratio,
    }
    wrapper = build_karman_2d_dit(cfg).to(device)
    n_params = sum(p.numel() for p in wrapper.parameters())
    print(f'Model params: {n_params:,}  |  flat_dim={FLAT_MAIN}  cond_dim={FLAT_COND}',
          flush=True)

    ema = deepcopy(wrapper).to(device)
    ema.eval()
    update_ema(ema.parameters(), wrapper.parameters(), rate=0)

    # ── Diffusion ──────────────────────────────────────────────────────────
    diffusion_train = create_diffusion(
        timestep_respacing='',
        noise_schedule=args.noise_schedule,
        learn_sigma=False,
        diffusion_steps=args.T_diffusion,
    )
    diffusion_sample = create_diffusion(
        timestep_respacing=str(args.sample_steps),
        noise_schedule=args.noise_schedule,
        learn_sigma=False,
        diffusion_steps=args.T_diffusion,
    )

    opt     = torch.optim.AdamW(wrapper.parameters(), lr=args.lr, weight_decay=0)
    use_amp = args.mixed_precision and device.type == 'cuda'
    print(f'Mixed precision: {"bfloat16" if use_amp else "off"}', flush=True)

    # ── Resume ─────────────────────────────────────────────────────────────
    step = 0; start_epoch = 0
    resume_path = args.resume
    if resume_path is None:
        existing = sorted(glob.glob(os.path.join(ckpt_dir, '*.pt')))
        if existing:
            resume_path = existing[-1]
            print(f'Auto-resume: {resume_path}', flush=True)
    if resume_path:
        print(f'Loading checkpoint {resume_path} ...', flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw  = wrapper._orig_mod if hasattr(wrapper, '_orig_mod') else wrapper
        sd   = ckpt['model']
        sd   = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
        raw.load_state_dict(sd)
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['opt'])
        step        = ckpt['step']
        start_epoch = ckpt.get('epoch', 0)
        print(f'  Resumed at step={step}  epoch={start_epoch}', flush=True)

    # ── WandB ──────────────────────────────────────────────────────────────
    wandb.init(
        project = args.wandb_project,
        name    = args.wandb_run,
        entity  = args.wandb_entity,
        config  = vars(args),
        dir     = args.outdir,
    )
    wandb.config.update({'n_params': n_params})

    # ── Resolve total steps ────────────────────────────────────────────────
    steps_per_epoch = len(train_dataset) // args.batch_size
    if args.n_epochs is not None:
        total_steps = args.n_epochs * steps_per_epoch
        print(f'n_epochs={args.n_epochs}  steps_per_epoch={steps_per_epoch}  '
              f'total_steps={total_steps}', flush=True)
    else:
        total_steps = args.n_steps
        print(f'total_steps={total_steps}  '
              f'(≈ {total_steps / max(steps_per_epoch, 1):.1f} epochs)', flush=True)

    # ── Training loop ──────────────────────────────────────────────────────
    wrapper.train()
    running_loss = 0.0; log_steps = 0; train_start = time(); done = False
    epoch = start_epoch

    while not done:
        pbar = tqdm(loader, desc=f'epoch {epoch}', leave=True)
        for batch in pbar:
            x_flat, cond_flat, niu, cx, cy, r, re = pack(batch)

            ts = torch.randint(0, diffusion_train.num_timesteps,
                               (x_flat.shape[0],), device=device)

            with autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss_dict = diffusion_train.training_losses(
                    wrapper, x_flat, ts,
                    model_kwargs={'cond_flat': cond_flat,
                                  'niu': niu, 'cx': cx, 'cy': cy, 'r': r, 're': re},
                )
                loss = loss_dict['loss'].mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), args.grad_clip)
            opt.step()
            update_ema(ema.parameters(), wrapper.parameters(), rate=args.ema_rate)

            running_loss += loss.item(); log_steps += 1; step += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}', step=step, epoch=epoch)

            if step % args.log_every == 0:
                avg_loss      = running_loss / log_steps
                elapsed_hours = (time() - train_start) / 3600
                mem_gb        = (torch.cuda.max_memory_allocated(device) / 1024 ** 3
                                 if device.type == 'cuda' else 0.0)
                print(f'step={step:07d}  epoch={epoch}  loss={avg_loss:.4f}  '
                      f'elapsed={elapsed_hours:.3f}h  mem={mem_gb:.2f}GB', flush=True)
                wandb.log({
                    'train/loss':          avg_loss,
                    'train/epoch':         epoch,
                    'train/elapsed_hours': elapsed_hours,
                    'train/peak_mem_gb':   mem_gb,
                }, step=step)
                running_loss = 0.0; log_steps = 0
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)

            if step >= total_steps:
                done = True; break

        epoch += 1

        is_last = done
        if epoch % args.vis_every_epochs == 0 or is_last:
            raw_ema = ema._orig_mod if hasattr(ema, '_orig_mod') else ema
            log_dict = generate_and_visualize(
                raw_ema, diffusion_sample, test_dataset, device,
                train_dataset, n_vis, args.n_frames_vis, step, epoch, vis_dir,
            )
            wandb.log({'train/epoch': epoch, **log_dict}, step=step)

        _MILESTONE_EPOCHS = {100, 200, 500, 1000, 1500, 2000}
        if epoch % args.save_every_epochs == 0 or is_last or epoch in _MILESTONE_EPOCHS:
            ckpt_path = os.path.join(ckpt_dir, f'epoch{epoch:05d}_step{step:07d}.pt')
            raw_w = wrapper._orig_mod if hasattr(wrapper, '_orig_mod') else wrapper
            torch.save({
                'model': raw_w.state_dict(),
                'ema':   ema.state_dict(),
                'opt':   opt.state_dict(),
                'step':  step,
                'epoch': epoch,
                'cfg':   cfg,
                'args':  vars(args),
            }, ckpt_path)
            print(f'Saved checkpoint → {ckpt_path}', flush=True)

            def _ckpt_epoch(p):
                try:
                    return int(os.path.basename(p).split('_')[0].replace('epoch', ''))
                except (ValueError, IndexError):
                    return -1

            all_ckpts     = sorted(glob.glob(os.path.join(ckpt_dir, '*.pt')))
            regular_ckpts = [p for p in all_ckpts if _ckpt_epoch(p) not in _MILESTONE_EPOCHS]
            for old in regular_ckpts[:-1]:
                os.remove(old)
                print(f'  Removed old checkpoint: {old}', flush=True)

    wandb.finish()
    print(f'Training complete.  steps={step}  epochs={epoch}', flush=True)


if __name__ == '__main__':
    main()
