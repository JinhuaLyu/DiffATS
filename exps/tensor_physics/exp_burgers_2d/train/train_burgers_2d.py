"""
train_burgers_2d.py — Conditional DDPM training on 2D Burgers Tucker factors.

Tucker rank = [5, 20, 20].  Conditions:
  - Initial frame (U_ic, Vh_ic) as token-level context
  - Scalar viscosity nu (normalised log-nu) via ScalarEmbedder + AdaLN

Token layout (813 total):
  [COND (148) | MAIN (665)]  — only MAIN tokens are noised/denoised.

All hyperparameters live in a YAML config; CLI args override YAML values.

Usage:
    cd ${REPO_ROOT}/tensor_physics/exp_burgers_2d
    python train_burgers_2d.py --config configs/train_v1.yaml --device cuda:0

    # Smoke test:
    python train_burgers_2d.py --config configs/train_v1.yaml \\
        --n_epochs 2 --batch_size 4 --log_every 1 --wandb_run smoke --device cpu
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

# ── Repo paths ─────────────────────────────────────────────────────────────
_EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '${REPO_ROOT}/video')

from diffusion import create_diffusion
from models.nn import update_ema

from dataset_burgers_2d import BurgersTucker2DDataset, reconstruct_video
from model_burgers_2d_dit import (
    build_burgers_2d_dit,
    FLAT_MAIN, FLAT_COND,
    FLAT_U1, FLAT_U3, FLAT_G,
    R_T, R_W, T_DIM, H_DIM,
)


# ---------------------------------------------------------------------------
# Pack / unpack helpers
# ---------------------------------------------------------------------------

def pack(batch):
    """
    Pack dataset batch into (x_flat, cond_flat, nu_batch, cd_batch).

    x_flat    : (B, 16360) — [U1 | U3 | G] normalised
    cond_flat : (B,  5120) — [U_ic | Vh_ic] normalised
    nu_batch  : (B,)       — normalised log-nu
    cd_batch  : (B,)       — normalised convection_delta
    """
    U1    = batch['U1']     # (B, 200,   5)
    U3    = batch['U3']     # (B, 128,  20)
    G     = batch['G']      # (B,   5, 128, 20)
    U_ic  = batch['U_ic']  # (B, 128,  20)
    Vh_ic = batch['Vh_ic'] # (B,  20, 128)
    nu    = batch['nu']     # (B,)
    cd    = batch['cd']     # (B,)

    x_flat    = torch.cat([U1.flatten(1), U3.flatten(1), G.flatten(1)], dim=1)
    cond_flat = torch.cat([U_ic.flatten(1), Vh_ic.flatten(1)],          dim=1)
    return x_flat, cond_flat, nu, cd


def unpack_x(x_flat, B):
    """Return (U1, U3, G) from flat tensor (normalised)."""
    c0, c1, c2 = x_flat.split([FLAT_U1, FLAT_U3, FLAT_G], dim=1)
    U1 = c0.reshape(B, T_DIM, R_T)
    U3 = c1.reshape(B, H_DIM, R_W)
    G  = c2.reshape(B, R_T, H_DIM, R_W)
    return U1, U3, G


# ---------------------------------------------------------------------------
# 3D spatiotemporal heatmap via exponax (requires JAX + exponax)
# ---------------------------------------------------------------------------

def _render_to_array(video_np, vlim, resolution=384):
    """
    Render one (T, H, W) video to a numpy uint8 RGB array using exponax.
    vlim : (vmin, vmax) shared across all panels.
    Returns ndarray (h, w, 3) or None on failure.
    """
    import io
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    import exponax as ex
    from exponax.viz._volume import zigzag_alpha
    from functools import partial
    from PIL import Image as PILImage

    trj = jnp.array(video_np[::-1, None, :, :])   # (T, 1, H, W), reversed so T=0 faces viewer
    fig = ex.viz.plot_spatio_temporal_2d(
        trj, vlim=vlim, cmap='twilight',
        bg_color='white', resolution=resolution,
        transfer_function=partial(zigzag_alpha, min_alpha=0.05),
        gamma_correction=2.0,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=72, bbox_inches='tight')
    buf.seek(0)
    arr = np.array(PILImage.open(buf).convert('RGB'))
    plt.close(fig)
    return arr


def render_3d_comparison_grid(videos_real, videos_gen,
                               epoch, step, mean_rel_err, resolution=384):
    """
    Render all real/generated videos with a shared vlim, stitch into a
    B-row × 2-col grid (Real | Generated), add a shared twilight colorbar,
    and return the composite matplotlib Figure (or None if exponax fails).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        B = len(videos_real)
        rows = []
        for i in range(B):
            # Each cube uses its own 98th-percentile vlim
            vlim_real = float(np.percentile(np.abs(videos_real[i]), 98))
            vlim_gen  = float(np.percentile(np.abs(videos_gen[i]),  98))
            print(f'  sample {i}: real vlim=({-vlim_real:.4f},{vlim_real:.4f})  '
                  f'gen vlim=({-vlim_gen:.4f},{vlim_gen:.4f})')
            arr_real = _render_to_array(videos_real[i], (-vlim_real, vlim_real), resolution)
            arr_gen  = _render_to_array(videos_gen[i],  (-vlim_gen,  vlim_gen),  resolution)
            if arr_real is None or arr_gen is None:
                return None
            rows.append(np.concatenate([arr_real, arr_gen], axis=1))  # (h, 2w, 3)

        canvas = np.concatenate(rows, axis=0)   # (B*h, 2w, 3)
        h_tot, w_tot = canvas.shape[:2]

        fig_out, ax = plt.subplots(
            figsize=(w_tot / 72, h_tot / 72 + 0.6), dpi=72,
        )
        ax.imshow(canvas)
        ax.set_title('Real  |  Generated', fontsize=10)
        ax.axis('off')

        # ── T-direction arrow (depth axis projects ≈ upper-right in vape4d default view)
        cell_h = h_tot / B
        cell_w = w_tot / 2
        x0 = cell_w * 0.6
        y0 = cell_h * 0.85
        dx = cell_w * 0.20
        dy = -cell_h * 0.13
        ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
        ax.text(x0 + dx * 1.15, y0 + dy * 1.15, 't',
                fontsize=11, color='black', fontweight='bold',
                ha='center', va='center')

        # Colorbar shows colormap shape only (each cube is independently scaled)
        sm = ScalarMappable(cmap='twilight', norm=Normalize(-1, 1))
        sm.set_array([])
        cb = fig_out.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cb.set_ticks([])
        cb.set_label('per-cube scale', fontsize=8)
        fig_out.suptitle(
            f'epoch={epoch}  step={step}  rel_err={mean_rel_err:.4f}',
            fontsize=9, y=0.02,
        )
        fig_out.tight_layout()
        return fig_out
    except Exception as e:
        print(f'  [3D grid render skipped: {e}]')
        return None


# ---------------------------------------------------------------------------
# Visualization: generate n_vis samples, render 3D spatiotemporal heatmaps
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_and_visualize(
    wrapper, diffusion_sample, test_dataset, device,
    train_dataset, n_vis: int, step: int, epoch: int, vis_dir: str,
):
    """
    Randomly sample n_vis items from test_dataset, run DDIM, render composite
    3D spatiotemporal grid (B rows × 2 cols), upload as val/3d_comparison.
    Returns dict for wandb.log.
    """
    wrapper.eval()
    B = n_vis

    # Random subset of test indices (different each call)
    vis_indices = torch.randperm(len(test_dataset))[:B].tolist()
    samples_list = [test_dataset[i] for i in vis_indices]
    test_batch = {
        k: (torch.stack([s[k] for s in samples_list])
            if k != 'idx'
            else [s[k] for s in samples_list])
        for k in samples_list[0].keys()
    }

    x_flat, cond_flat, nu_batch, cd_batch = pack(test_batch)
    x_flat    = x_flat[:B].to(device)
    cond_flat = cond_flat[:B].to(device)
    nu_batch  = nu_batch[:B].to(device)
    cd_batch  = cd_batch[:B].to(device)

    noise   = torch.randn(B, FLAT_MAIN, device=device)
    samples = diffusion_sample.p_sample_loop(
        wrapper, noise.shape, noise=noise,
        clip_denoised=False,
        model_kwargs={'cond_flat': cond_flat, 'nu': nu_batch, 'cd': cd_batch},
        device=device, progress=False,
    )

    U1_gen,  U3_gen,  G_gen  = unpack_x(samples.float().cpu(), B)
    U1_real, U3_real, G_real = unpack_x(x_flat.float().cpu(),  B)

    rel_errors  = []
    videos_real = []
    videos_gen  = []
    log_dict    = {}

    for i in range(B):
        def _recon(U1_i, U3_i, G_i):
            U1_dn = train_dataset.denorm(U1_i, 'U1').cpu().numpy()
            U3_dn = train_dataset.denorm(U3_i, 'U3').cpu().numpy()
            G_dn  = train_dataset.denorm(G_i,  'G').cpu().numpy()
            return reconstruct_video(U1_dn, U3_dn, G_dn).astype(np.float32)

        v_real = _recon(U1_real[i], U3_real[i], G_real[i])
        v_gen  = _recon(U1_gen[i],  U3_gen[i],  G_gen[i])

        rel_err = float(np.linalg.norm(v_real - v_gen) /
                        (np.linalg.norm(v_real) + 1e-8))
        rel_errors.append(rel_err)
        videos_real.append(v_real)
        videos_gen.append(v_gen)

    mean_rel_err = float(np.mean(rel_errors))
    print(f'  rel_err: {[f"{e:.4f}" for e in rel_errors]}  mean={mean_rel_err:.4f}')

    # ── Composite 3D spatiotemporal heatmap (B rows × 2 cols) ─────────────
    fig_grid = render_3d_comparison_grid(
        videos_real, videos_gen, epoch, step, mean_rel_err,
    )
    if fig_grid is not None:
        import matplotlib.pyplot as plt
        log_dict['val/3d_comparison'] = wandb.Image(fig_grid)
        plt.close(fig_grid)

    log_dict['val/rel_error_mean'] = mean_rel_err
    wrapper.train()
    return log_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train 2D Burgers Tucker DiT')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config (absolute or relative to script dir)')
    # All args below can override YAML values
    parser.add_argument('--data_dir',          type=str,   default=None)
    parser.add_argument('--test_data_dir',     type=str,   default=None,
                        help='Separate directory for test Tucker factors. '
                             'If set, test dataset uses this dir and train uses all of data_dir.')
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

    # ── Defaults ───────────────────────────────────────────────────────────────
    defaults = dict(
        data_dir          = os.path.join(_EXP, 'data', 'tucker_burgers_rT5_rH20_rW20'),
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
        save_every_epochs = 20,
        n_test            = 20,
        n_vis             = 4,
        T_diffusion       = 1000,
        sample_steps      = 250,
        noise_schedule    = 'linear',
        ema_rate          = 0.9999,
        grad_clip         = 1.0,
        mixed_precision   = True,
        outdir            = os.path.join(_EXP, 'output', 'train'),
        wandb_project     = '<PROJECT>',
        wandb_run         = 'burgers2d_tucker_dit_v1',
        wandb_entity      = None,
        device            = 'auto',
        seed              = 0,
        resume            = None,
    )

    # ── Load YAML ──────────────────────────────────────────────────────────────
    if cli.config is not None:
        cfg_path = cli.config if os.path.isabs(cli.config) else \
                   os.path.join(_EXP, cli.config)
        with open(cfg_path) as f:
            yaml_cfg = yaml.safe_load(f)
        # resolve data_dir / test_data_dir / outdir relative to script dir if not absolute
        for key in ('data_dir', 'test_data_dir', 'outdir'):
            if key in yaml_cfg and yaml_cfg[key] and not os.path.isabs(yaml_cfg[key]):
                yaml_cfg[key] = os.path.normpath(os.path.join(_EXP, yaml_cfg[key]))
        defaults.update({k: v for k, v in yaml_cfg.items() if v is not None})

    # ── CLI overrides YAML ─────────────────────────────────────────────────────
    for key, val in vars(cli).items():
        if key == 'config':
            continue
        if val is not None:
            if key == 'mixed_precision':
                defaults[key] = val.lower() != 'false'
            else:
                defaults[key] = val

    args = argparse.Namespace(**defaults)

    # ── Setup ──────────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ckpt_dir = os.path.join(args.outdir, 'checkpoints')
    vis_dir  = os.path.join(args.outdir, 'vis')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir,  exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────────
    if args.test_data_dir is not None:
        # Separate test directory: train uses all of data_dir, test loaded
        # independently but reuses train's normalization stats
        train_dataset = BurgersTucker2DDataset(args.data_dir,
                                                split='all', device=device)
        test_dataset  = BurgersTucker2DDataset(args.test_data_dir,
                                                split='all', device=device,
                                                external_stats=train_dataset.stats)
    else:
        # Build full dataset first to determine N_total, then split last n_test
        full_ds = BurgersTucker2DDataset(args.data_dir, split='all', device=device)
        N_total = len(full_ds)
        print(f'Total samples: {N_total}')

        test_indices  = list(range(N_total - args.n_test, N_total))
        train_dataset = BurgersTucker2DDataset(args.data_dir,
                                                test_indices=test_indices,
                                                split='train', device=device)
        test_dataset  = BurgersTucker2DDataset(args.data_dir,
                                                test_indices=test_indices,
                                                split='test',  device=device)
    print(f'Train: {len(train_dataset)}  |  Test: {len(test_dataset)}')

    n_vis = min(args.n_vis, len(test_dataset))

    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    cfg = {
        'hidden_size': args.hidden_size,
        'depth':       args.depth,
        'num_heads':   args.num_heads,
        'mlp_ratio':   args.mlp_ratio,
    }
    wrapper = build_burgers_2d_dit(cfg).to(device)
    n_params = sum(p.numel() for p in wrapper.parameters())
    print(f'Model params: {n_params:,}  |  flat_dim={FLAT_MAIN}  cond_dim={FLAT_COND}')

    ema = deepcopy(wrapper).to(device)
    ema.eval()
    update_ema(ema.parameters(), wrapper.parameters(), rate=0)

    # ── Diffusion ──────────────────────────────────────────────────────────────
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

    # ── Optimizer ──────────────────────────────────────────────────────────────
    opt    = torch.optim.AdamW(wrapper.parameters(), lr=args.lr, weight_decay=0)
    use_amp = args.mixed_precision and device.type == 'cuda'
    print(f'Mixed precision: {"bfloat16" if use_amp else "off"}')

    # ── Resume ─────────────────────────────────────────────────────────────────
    step = 0; start_epoch = 0
    resume_path = args.resume
    if resume_path is None:
        existing = sorted(glob.glob(os.path.join(ckpt_dir, '*.pt')))
        if existing:
            resume_path = existing[-1]
            print(f'Auto-resume: {resume_path}')
    if resume_path:
        print(f'Loading checkpoint {resume_path} ...')
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw  = wrapper._orig_mod if hasattr(wrapper, '_orig_mod') else wrapper
        sd   = ckpt['model']
        sd   = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
        raw.load_state_dict(sd)
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['opt'])
        step        = ckpt['step']
        start_epoch = ckpt.get('epoch', 0)
        print(f'  Resumed at step={step}  epoch={start_epoch}')

    # ── WandB ──────────────────────────────────────────────────────────────────
    wandb.init(
        project = args.wandb_project,
        name    = args.wandb_run,
        entity  = args.wandb_entity,
        config  = vars(args),
        dir     = args.outdir,
    )
    wandb.config.update({'n_params': n_params})

    # ── Resolve total steps ────────────────────────────────────────────────────
    steps_per_epoch = len(train_dataset) // args.batch_size
    if args.n_epochs is not None:
        total_steps = args.n_epochs * steps_per_epoch
        print(f'n_epochs={args.n_epochs}  steps_per_epoch={steps_per_epoch}  '
              f'total_steps={total_steps}')
    else:
        total_steps = args.n_steps
        print(f'total_steps={total_steps}  '
              f'(≈ {total_steps / max(steps_per_epoch, 1):.1f} epochs)')

    # ── Training loop ──────────────────────────────────────────────────────────
    wrapper.train()
    running_loss = 0.0; log_steps = 0; train_start = time(); done = False
    epoch = start_epoch

    while not done:
        pbar = tqdm(loader, desc=f'epoch {epoch}', leave=True)
        for batch in pbar:
            x_flat, cond_flat, nu_batch, cd_batch = pack(batch)

            ts = torch.randint(0, diffusion_train.num_timesteps,
                               (x_flat.shape[0],), device=device)

            with autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss_dict = diffusion_train.training_losses(
                    wrapper, x_flat, ts,
                    model_kwargs={'cond_flat': cond_flat, 'nu': nu_batch, 'cd': cd_batch},
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

            # ── Periodic scalar logging ────────────────────────────────────────
            if step % args.log_every == 0:
                avg_loss      = running_loss / log_steps
                elapsed_hours = (time() - train_start) / 3600
                mem_gb        = (torch.cuda.max_memory_allocated(device) / 1024 ** 3
                                 if device.type == 'cuda' else 0.0)
                print(f'step={step:07d}  epoch={epoch}  loss={avg_loss:.4f}  '
                      f'elapsed={elapsed_hours:.3f}h  mem={mem_gb:.2f}GB')
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

        # ── Epoch-end: visualise + checkpoint ─────────────────────────────────
        is_last = done
        if epoch % args.vis_every_epochs == 0 or is_last:
            raw_ema = ema._orig_mod if hasattr(ema, '_orig_mod') else ema
            log_dict = generate_and_visualize(
                raw_ema, diffusion_sample, test_dataset, device,
                train_dataset, n_vis, step, epoch, vis_dir,
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
            print(f'Saved checkpoint → {ckpt_path}')

            # Milestone checkpoints are never deleted; among the rest keep only latest 1
            def _ckpt_epoch(p):
                try:
                    return int(os.path.basename(p).split('_')[0].replace('epoch', ''))
                except (ValueError, IndexError):
                    return -1

            all_ckpts    = sorted(glob.glob(os.path.join(ckpt_dir, '*.pt')))
            regular_ckpts = [p for p in all_ckpts if _ckpt_epoch(p) not in _MILESTONE_EPOCHS]
            for old in regular_ckpts[:-1]:
                os.remove(old)
                print(f'  Removed old checkpoint: {old}')

    wandb.finish()
    print(f'Training complete.  steps={step}  epochs={epoch}')


if __name__ == '__main__':
    main()
