"""
compute_fvd.py -- Frechet Video Distance between two video .pt tensors.

Uses the StyleGAN-V TorchScript I3D (Kinetics-400 pre-trained), which is the
de-facto reference detector for FVD in the generative-video literature.

Input tensor conventions (auto-detected):
    (T, N, H, W)  uint8   -- standard for this project
    (N, T, H, W)  uint8   -- also accepted

Grayscale frames are replicated to RGB. Frames are resized to 224x224 and
normalized to [-1, 1] inside the I3D TorchScript via rescale=True, resize=True.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.linalg import sqrtm


def load_video_tensor(path, key='videos', n=None):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict):
        x = obj[key]
    else:
        x = obj
    assert x.dtype == torch.uint8, f'{path}: expected uint8, got {x.dtype}'
    if x.dim() != 4:
        raise ValueError(f'{path}: expected 4D tensor, got shape {tuple(x.shape)}')
    # Normalize layout to (N, T, H, W)
    if x.shape[0] < x.shape[1] and x.shape[0] <= 64:
        # (T, N, H, W) -- first dim is small (frames), second is large (videos)
        x = x.permute(1, 0, 2, 3).contiguous()
        layout_src = '(T, N, H, W)'
    else:
        layout_src = '(N, T, H, W)'
    if n is not None:
        x = x[:n]
    print(f'loaded {path}  from {layout_src}  -> (N={x.shape[0]}, T={x.shape[1]}, '
          f'H={x.shape[2]}, W={x.shape[3]})  dtype={x.dtype}')
    return x


@torch.no_grad()
def extract_features(videos_NTHW, model, device, batch_size=16):
    """videos_NTHW: (N, T, H, W) uint8. Returns (N, D) float32 features (cpu)."""
    N, T, H, W = videos_NTHW.shape
    # Probe output dim once with a single sample (positional args; TorchScript
    # signature: forward(x, rescale, resize, return_features))
    probe_x = videos_NTHW[:1].to(device).float().unsqueeze(1).expand(1, 3, T, H, W)
    probe_out = model(probe_x, True, True, True)
    D = probe_out.shape[1]
    print(f'  I3D feature dim = {D}', flush=True)
    feats = torch.empty((N, D), dtype=torch.float32)
    t0 = time.time()
    for i in range(0, N, batch_size):
        batch = videos_NTHW[i:i + batch_size]
        b = batch.shape[0]
        x = batch.to(device).float()
        x = x.unsqueeze(1).expand(b, 3, T, H, W)
        # Positional args: (x, rescale=True, resize=True, return_features=True)
        f = model(x, True, True, True)
        feats[i:i + b] = f.float().cpu()
        if (i // batch_size) % 20 == 0:
            elapsed = time.time() - t0
            print(f'  feats {i + b:5d}/{N}  elapsed={elapsed:.1f}s', flush=True)
    print(f'  feats done  total={time.time() - t0:.1f}s', flush=True)
    return feats


def compute_fvd_from_features(feats_r, feats_g):
    """Frechet distance between two Gaussians fit to feats_r and feats_g."""
    mu_r = feats_r.mean(axis=0)
    mu_g = feats_g.mean(axis=0)
    # rowvar=False: rows are observations, columns are features
    sigma_r = np.cov(feats_r, rowvar=False)
    sigma_g = np.cov(feats_g, rowvar=False)

    diff = mu_r - mu_g
    # sqrtm of matrix product (can be complex due to tiny numerical errors)
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        # If imaginary part is small, discard it
        if np.max(np.abs(covmean.imag)) > 1e-3:
            print(f'  WARNING: sqrtm imag max={np.max(np.abs(covmean.imag)):.3e}')
        covmean = covmean.real

    fvd = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    return fvd, float(np.linalg.norm(diff)**2), float(np.trace(sigma_r)), float(np.trace(sigma_g))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--real',     required=True, type=str)
    p.add_argument('--gen',      required=True, type=str)
    p.add_argument('--n',        type=int, default=10000)
    p.add_argument('--batch',    type=int, default=16)
    p.add_argument('--i3d',      type=str,
                   default='${HOME}/.cache/fvd/i3d_torchscript.pt')
    p.add_argument('--real_key', type=str, default='videos',
                   help='Dict key if --real is a dict .pt. Ignored for bare tensors.')
    p.add_argument('--gen_key',  type=str, default='videos')
    p.add_argument('--out',      type=str, required=True,
                   help='Output JSON path for the report.')
    p.add_argument('--real_select', choices=['first', 'random'], default='first')
    p.add_argument('--seed',     type=int, default=0)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading I3D: {args.i3d}')
    model = torch.jit.load(args.i3d, map_location=device).eval()

    real = load_video_tensor(args.real, key=args.real_key)
    gen  = load_video_tensor(args.gen,  key=args.gen_key, n=args.n)

    if args.real_select == 'random':
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(real.shape[0], generator=g)[:args.n]
        real = real[idx]
        print(f'real: random {args.n} samples, seed={args.seed}')
    else:
        real = real[:args.n]
        print(f'real: first {args.n} samples')

    assert real.shape[1:] == gen.shape[1:], \
        f'shape mismatch: real {tuple(real.shape)}, gen {tuple(gen.shape)}'
    assert real.shape[0] == args.n and gen.shape[0] == args.n

    print('\n[1/3] Extracting real features...')
    feats_r = extract_features(real, model, device, args.batch).numpy()
    print('\n[2/3] Extracting generated features...')
    feats_g = extract_features(gen,  model, device, args.batch).numpy()

    print('\n[3/3] Computing FVD...')
    fvd, mean_sq, tr_r, tr_g = compute_fvd_from_features(feats_r, feats_g)
    print(f'\n=== FVD = {fvd:.4f} ===')
    print(f'  ||mu_r - mu_g||^2 = {mean_sq:.4f}')
    print(f'  tr(Sigma_r)       = {tr_r:.4f}')
    print(f'  tr(Sigma_g)       = {tr_g:.4f}')

    report = {
        'fvd': fvd,
        'mean_sq_diff': mean_sq,
        'trace_sigma_real': tr_r,
        'trace_sigma_gen': tr_g,
        'n_samples': int(args.n),
        'frame_count': int(real.shape[1]),
        'frame_size': [int(real.shape[2]), int(real.shape[3])],
        'batch_size': int(args.batch),
        'real_path': os.path.abspath(args.real),
        'gen_path': os.path.abspath(args.gen),
        'real_select': args.real_select,
        'real_seed': int(args.seed) if args.real_select == 'random' else None,
        'i3d_ckpt': os.path.abspath(args.i3d),
        'feature_dim': int(feats_r.shape[1]),
        'device': str(device),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Report saved: {args.out}')


if __name__ == '__main__':
    main()
