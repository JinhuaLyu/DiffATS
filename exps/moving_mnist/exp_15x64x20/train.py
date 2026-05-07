"""
train.py -- Diffusion training on Tucker 3-part factors (G stored directly).

Tucker factors per video (G = einsum('ijk,hj->ihk', C, U_2), pre-computed):
  U_1 : (T=20,  r_T)
  G   : (r_T,   H=64, r_W)  ->  stored flat as (r_T*H, r_W)
  U_3 : (W=64,  r_W)

Flat vector: [U_1 | U_3 | G]  dim = T*r_T + W*r_W + r_T*H*r_W

Usage:
    cd ${REPO_ROOT}/exps/moving_mnist/exp_15x64x20
    python train.py train.yaml
    python train.py train.yaml \\
        --max_steps 10 --batch_size 4 --compile false   # sanity check
"""

import argparse
import glob
import os
import sys
import yaml
from copy import deepcopy
from time import time

import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.nn import update_ema
from diffusion import create_diffusion

_EXP_DIR        = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_EXP_DIR, 'train.yaml')


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_cli(cfg: dict, cli_args) -> dict:
    for key, val in vars(cli_args).items():
        if key == 'config':
            continue
        if val is not None:
            cfg[key] = val
    return cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShardedTuckerGDataset(Dataset):
    """
    Loads Tucker factor shards produced by save_tucker_slow_g.py.
    Shard keys: U_1, G, U_3  (G is pre-computed and stored directly).

    Returns (U_1, U_3, G_flat) where G_flat = G.reshape(r_T*H, r_W).
    """

    def __init__(self, shard_dir: str, preload: bool = True):
        pattern = os.path.join(shard_dir, 'tucker_factors_shard_*.pt')
        self.paths = sorted(glob.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f'No Tucker shard files found in {shard_dir}')
        self._meta = []
        self._data = {}
        for i, p in enumerate(self.paths):
            d = torch.load(p, map_location='cpu', weights_only=False)
            n = d['U_1'].shape[0]
            for j in range(n):
                self._meta.append((i, j))
            if preload:
                self._data[i] = d

    def __len__(self):
        return len(self._meta)

    def __getitem__(self, idx):
        si, li = self._meta[idx]
        if si not in self._data:
            self._data[si] = torch.load(
                self.paths[si], map_location='cpu', weights_only=False)
        d = self._data[si]
        U_1 = d['U_1'][li].float()    # (T,  r_T)
        G   = d['G'][li].float()      # (r_T, H, r_W)
        U_3 = d['U_3'][li].float()    # (W,  r_W)
        r_T, H, r_W = G.shape
        G_flat = G.reshape(r_T * H, r_W)   # (r_T*H, r_W)
        return U_1, U_3, G_flat


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

def compute_stats(shard_dir: str, force: bool = False) -> dict:
    stats_path = os.path.join(shard_dir, 'tucker_g_stats.pt')
    if os.path.exists(stats_path) and not force:
        print(f'Loading cached stats from {stats_path}')
        return torch.load(stats_path, map_location='cpu', weights_only=False)

    print('Computing G-parts stats (one pass over all shards)...')
    sums  = {'U1': 0., 'U3': 0., 'G': 0.}
    sums2 = {'U1': 0., 'U3': 0., 'G': 0.}
    cnts  = {'U1': 0,  'U3': 0,  'G': 0}

    for p in tqdm(sorted(glob.glob(os.path.join(shard_dir, 'tucker_factors_shard_*.pt'))),
                  desc='stats'):
        d = torch.load(p, map_location='cpu', weights_only=False)
        U_1_batch = d['U_1'].float()   # (B, T, r_T)
        G_batch   = d['G'].float()     # (B, r_T, H, r_W)
        U_3_batch = d['U_3'].float()   # (B, W, r_W)

        for key, tensor in [('U1', U_1_batch), ('G', G_batch), ('U3', U_3_batch)]:
            sums[key]  += tensor.sum().item()
            sums2[key] += (tensor * tensor).sum().item()
            cnts[key]  += tensor.numel()

    stats = {}
    for key in ('U1', 'U3', 'G'):
        mean = sums[key] / cnts[key]
        std  = float((sums2[key] / cnts[key] - mean ** 2) ** 0.5)
        stats[f'std_{key}'] = std
        print(f'  std_{key} = {std:.6f}')

    torch.save(stats, stats_path)
    print(f'Saved stats -> {stats_path}')
    return stats


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------

def normalize_and_pack(U_1, U_3, G_flat, stats):
    return torch.cat([
        (U_1    / stats['std_U1']).flatten(1),
        (U_3    / stats['std_U3']).flatten(1),
        (G_flat / stats['std_G']).flatten(1),
    ], dim=1)


def unpack_and_denormalize(x_flat, stats, split, shapes, device):
    B = x_flat.shape[0]
    chunks = x_flat.split(split, dim=1)
    U_1    = chunks[0].reshape(B, *shapes[0]) * stats['std_U1'].to(device)
    U_3    = chunks[1].reshape(B, *shapes[1]) * stats['std_U3'].to(device)
    G_flat = chunks[2].reshape(B, *shapes[2]) * stats['std_G'].to(device)
    return U_1, U_3, G_flat


# ---------------------------------------------------------------------------
# Video reconstruction
# ---------------------------------------------------------------------------

def reconstruct_videos(U_1, U_3, G_flat, r_T, H, r_W):
    """
    U_1    : (B, T,  r_T)
    U_3    : (B, W,  r_W)
    G_flat : (B, r_T*H, r_W)  ->  reshape to (B, r_T, H, r_W)

    video[b,t,h,w] = Sigma_{a,c} G[a,h,c] * U_1[t,a] * U_3[w,c]
    """
    B = U_1.shape[0]
    G = G_flat.reshape(B, r_T, H, r_W)
    return torch.einsum('bahc,bta,bwc->bthw', G, U_1, U_3)


def make_sample_grid(videos_np: np.ndarray,
                     show_frames=tuple(range(20))) -> Image.Image:
    """Always outputs 64x64 frames x 20 cols per video row."""
    B, T, H, W = videos_np.shape
    frames = [t for t in show_frames if t < T]
    canvas = np.zeros((B * H, len(frames) * W), dtype=np.uint8)
    for b in range(B):
        for col, t in enumerate(frames):
            canvas[b*H:(b+1)*H, col*W:(col+1)*W] = videos_np[b, t]
    return Image.fromarray(canvas, mode='L')


@torch.no_grad()
def sample_and_visualize(wrapper, diffusion_sample, n, device, stats, cfg,
                         split, shapes, step, sample_dir):
    wrapper.eval()
    x_flat = torch.randn(n, sum(split), device=device)
    samples = diffusion_sample.p_sample_loop(
        wrapper, x_flat.shape, noise=x_flat,
        clip_denoised=False, model_kwargs={}, device=device, progress=False,
    )
    U_1, U_3, G_flat = unpack_and_denormalize(
        samples.float().cpu(), stats, split, shapes, 'cpu')
    r_T, H, r_W = cfg['r_T'], cfg['H'], cfg['r_W']
    videos    = reconstruct_videos(U_1, U_3, G_flat, r_T, H, r_W)
    videos_u8 = videos.clamp(0, 255).byte().numpy()   # (B, T=20, H=64, W=64)

    grid_pil  = make_sample_grid(videos_u8)
    grid_path = os.path.join(sample_dir, f'{step:07d}_grid.png')
    grid_pil.save(grid_path)

    gif_frames = [Image.fromarray(videos_u8[0, t], mode='L')
                  for t in range(videos_u8.shape[1])]
    gif_path = os.path.join(sample_dir, f'{step:07d}_sample0.gif')
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=80, loop=0)

    wrapper.train()
    return grid_pil, grid_path, gif_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict):
    from model import build_tucker_3parts_dit

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg['seed'])

    results_dir = os.path.join(_EXP_DIR, cfg['results_dir'])
    shard_dir   = cfg['shard_dir']
    ckpt_dir    = os.path.join(results_dir, 'checkpoints')
    sample_dir  = os.path.join(results_dir, 'samples')
    os.makedirs(ckpt_dir,   exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    wandb.init(
        project = cfg['wandb_project'],
        name    = cfg['wandb_run_name'] or None,
        entity  = cfg['wandb_entity']  or None,
        config  = cfg,
        dir     = results_dir,
    )

    raw_stats = compute_stats(shard_dir)
    stats = {k: torch.tensor(v, dtype=torch.float32, device=device)
             for k, v in raw_stats.items()}

    dataset   = ShardedTuckerGDataset(shard_dir, preload=True)
    n_workers = cfg['num_workers']
    loader    = DataLoader(
        dataset, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=n_workers, pin_memory=device.type == 'cuda',
        drop_last=True, persistent_workers=n_workers > 0,
        prefetch_factor=cfg.get('prefetch_factor', 2) if n_workers > 0 else None,
    )
    print(f'Dataset: {len(dataset):,} videos  |  Steps/epoch: {len(loader)}')

    wrapper  = build_tucker_3parts_dit(cfg).to(device)
    split    = wrapper._split
    shapes   = wrapper._shapes
    n_params = sum(p.numel() for p in wrapper.parameters())
    print(f'Tucker3PartsWrapper  params={n_params:,}')
    print(f'  seq_len={wrapper.core.seq_len}  flat_dim={sum(split)}')
    print(f'  split={split}')
    wandb.config.update({'n_params': n_params})

    ema = deepcopy(wrapper).to(device)
    ema.eval()
    update_ema(ema.parameters(), wrapper.parameters(), rate=0)

    if cfg.get('compile', False):
        print('torch.compile ...')
        wrapper = torch.compile(wrapper, mode='default')
        print('Done.')

    diffusion_train = create_diffusion(
        timestep_respacing='', noise_schedule=cfg['noise_schedule'],
        learn_sigma=False, diffusion_steps=cfg['diffusion_steps'],
    )
    diffusion_sample = create_diffusion(
        timestep_respacing=str(cfg['sample_steps']),
        noise_schedule=cfg['noise_schedule'],
        learn_sigma=False, diffusion_steps=cfg['diffusion_steps'],
    )

    opt     = torch.optim.AdamW(wrapper.parameters(), lr=cfg['lr'], weight_decay=0)
    use_amp = cfg.get('mixed_precision', False) and device.type == 'cuda'
    print(f'Mixed precision: {"bfloat16" if use_amp else "off"}')

    max_steps  = cfg.get('max_steps')
    max_epochs = cfg.get('max_epochs')

    step = 0; epoch = 0
    resume_path = cfg.get('resume')
    if resume_path is None:
        existing = sorted(glob.glob(os.path.join(ckpt_dir, '*.pt')))
        if existing:
            resume_path = existing[-1]
            print(f'Auto-detected checkpoint: {resume_path}')
    if resume_path:
        print(f'Resuming from {resume_path} ...')
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw  = wrapper._orig_mod if hasattr(wrapper, '_orig_mod') else wrapper
        model_sd = ckpt['model']
        if any(k.startswith('_orig_mod.') for k in model_sd):
            model_sd = {k.replace('_orig_mod.', '', 1): v for k, v in model_sd.items()}
        raw.load_state_dict(model_sd)
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['opt'])
        step  = ckpt['step']
        epoch = ckpt['epoch']
        print(f'  Resumed at epoch={epoch}  step={step}')

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    wrapper.train()
    running_loss = 0.0; log_steps = 0; t0 = time(); train_start = time(); done = False

    while not done:
        pbar = tqdm(loader, desc=f'epoch {epoch}', leave=True)
        for U_1_b, U_3_b, G_b in pbar:
            U_1_b = U_1_b.to(device, non_blocking=True)
            U_3_b = U_3_b.to(device, non_blocking=True)
            G_b   = G_b.to(device,   non_blocking=True)

            x_flat = normalize_and_pack(U_1_b, U_3_b, G_b, stats)
            ts     = torch.randint(0, diffusion_train.num_timesteps,
                                   (x_flat.shape[0],), device=device)

            with autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss_dict = diffusion_train.training_losses(wrapper, x_flat, ts)
                loss      = loss_dict['loss'].mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), cfg['grad_clip'])
            opt.step()
            update_ema(ema.parameters(), wrapper.parameters(), rate=cfg['ema_rate'])

            running_loss += loss.item(); log_steps += 1; step += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}', step=step)

            if step % cfg['log_every'] == 0:
                avg_loss      = running_loss / log_steps
                elapsed       = time() - t0
                steps_sec     = log_steps / elapsed
                elapsed_hours = (time() - train_start) / 3600
                mem_gb = (torch.cuda.max_memory_allocated(device) / 1024 ** 3
                          if device.type == 'cuda' else 0.0)
                print(f'epoch={epoch:04d}  step={step:07d}  '
                      f'loss={avg_loss:.4f}  steps/s={steps_sec:.2f}  '
                      f'elapsed={elapsed_hours:.2f}h  peak_mem={mem_gb:.2f}GB')
                wandb.log({'train/loss': avg_loss,
                           'train/elapsed_hours': elapsed_hours,
                           'train/peak_mem_gb': mem_gb,
                           'train/epoch': epoch}, step=step)
                running_loss = 0.0; log_steps = 0; t0 = time()

            if max_steps is not None and step >= max_steps:
                done = True; break

        epoch += 1
        wandb.log({'train/epoch': epoch}, step=step)

        _milestone_epochs = {500, 1000, 1500, 2000}
        _milestone_tags   = {f'epoch{e:04d}' for e in _milestone_epochs}
        raw_wrapper = wrapper._orig_mod if hasattr(wrapper, '_orig_mod') else wrapper

        if epoch % cfg['ckpt_every_epoch'] == 0:
            ckpt_path = os.path.join(ckpt_dir, f'epoch{epoch:04d}.pt')
            torch.save({
                'model': raw_wrapper.state_dict(), 'ema': ema.state_dict(),
                'opt': opt.state_dict(), 'step': step, 'epoch': epoch,
                'cfg': cfg, 'stats': raw_stats,
            }, ckpt_path)
            print(f'Saved checkpoint -> {ckpt_path}')

            all_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pt')))
            regular   = [p for p in all_ckpts
                         if not any(tag in os.path.basename(p) for tag in _milestone_tags)]
            for old in regular[:-2]:
                os.remove(old)

            raw_ema = ema._orig_mod if hasattr(ema, '_orig_mod') else ema
            grid_pil, grid_path, gif_path = sample_and_visualize(
                raw_ema, diffusion_sample, n=cfg['n_samples'],
                device=device, stats=stats, cfg=cfg,
                split=split, shapes=shapes,
                step=step, sample_dir=sample_dir,
            )
            print(f'Saved samples -> {grid_path}')
            wandb.log({'samples': wandb.Image(
                grid_pil,
                caption=f'epoch {epoch} step {step} (EMA, {cfg["sample_steps"]} steps)',
            )}, step=step)

        if epoch in _milestone_epochs and epoch % cfg['ckpt_every_epoch'] != 0:
            milestone_path = os.path.join(ckpt_dir, f'epoch{epoch:04d}.pt')
            torch.save({
                'model': raw_wrapper.state_dict(), 'ema': ema.state_dict(),
                'opt': opt.state_dict(), 'step': step, 'epoch': epoch,
                'cfg': cfg, 'stats': raw_stats,
            }, milestone_path)
            print(f'Saved milestone checkpoint -> {milestone_path}')

        if max_epochs is not None and epoch >= max_epochs:
            done = True

    wandb.finish()
    print(f'Training complete. epochs={epoch}  steps={step}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, nargs='?', default=_DEFAULT_CONFIG)
    parser.add_argument('--resume',         type=str,   default=None)
    parser.add_argument('--shard_dir',      type=str,   default=None)
    parser.add_argument('--hidden_size',    type=int,   default=None)
    parser.add_argument('--depth',          type=int,   default=None)
    parser.add_argument('--num_heads',      type=int,   default=None)
    parser.add_argument('--batch_size',     type=int,   default=None)
    parser.add_argument('--max_steps',      type=int,   default=None)
    parser.add_argument('--max_epochs',     type=int,   default=None)
    parser.add_argument('--lr',             type=float, default=None)
    parser.add_argument('--results_dir',    type=str,   default=None)
    parser.add_argument('--wandb_run_name', type=str,   default=None)
    parser.add_argument('--log_every',      type=int,   default=None)
    parser.add_argument('--ckpt_every_epoch', type=int, default=None)
    parser.add_argument('--n_samples',      type=int,   default=None)
    parser.add_argument('--num_workers',    type=int,   default=None)
    parser.add_argument('--compile',
        type=lambda x: x.lower() != 'false', default=None)
    parser.add_argument('--mixed_precision',
        type=lambda x: x.lower() != 'false', default=None)
    args = parser.parse_args()
    cfg  = load_config(args.config)
    cfg  = merge_cli(cfg, args)
    main(cfg)
