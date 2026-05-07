"""
Generate N videos with a configurable number of sampling steps and save, for
each video, a single PNG containing 5 evenly-spaced frames laid out in one row.
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion import create_diffusion
from model import build_tucker_3parts_dit
from train import unpack_and_denormalize, reconstruct_videos


FLAT_SPLIT = [300, 1280, 19200]
FLAT_SHAPES = [(20, 15), (64, 20), (960, 20)]


def select_frames(T, k=5):
    return np.linspace(0, T - 1, k).round().astype(int).tolist()


def save_row_png(frames_u8, path):
    # frames_u8: list of (H, W) uint8 arrays
    row = np.concatenate(list(frames_u8), axis=1)
    Image.fromarray(row, mode='L').save(path)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, type=str)
    p.add_argument('--n_videos', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=10)
    p.add_argument('--sample_steps', type=int, default=1000)
    p.add_argument('--n_frames', type=int, default=5)
    p.add_argument('--outdir', required=True, type=str)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt['cfg']
    raw_stats = ckpt['stats']
    epoch = ckpt['epoch']
    print(f'Checkpoint epoch={epoch} step={ckpt["step"]} '
          f'noise_schedule={cfg["noise_schedule"]} diffusion_steps={cfg["diffusion_steps"]}')

    wrapper = build_tucker_3parts_dit(cfg).to(device)
    wrapper.load_state_dict(ckpt['ema'])
    wrapper.eval()

    diffusion_sample = create_diffusion(
        timestep_respacing=str(args.sample_steps),
        noise_schedule=cfg['noise_schedule'],
        learn_sigma=False,
        diffusion_steps=cfg['diffusion_steps'],
    )
    print(f'Sampler steps={args.sample_steps}')

    stats = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw_stats.items()}
    os.makedirs(args.outdir, exist_ok=True)

    flat_dim = sum(FLAT_SPLIT)
    produced = 0
    while produced < args.n_videos:
        bs = min(args.batch_size, args.n_videos - produced)
        x_flat = torch.randn(bs, flat_dim, device=device)
        samples = diffusion_sample.p_sample_loop(
            wrapper, x_flat.shape, noise=x_flat,
            clip_denoised=False, model_kwargs={},
            device=device, progress=True,
        )
        U_1, U_3, G_flat = unpack_and_denormalize(
            samples.float().cpu(), stats, FLAT_SPLIT, FLAT_SHAPES, 'cpu')
        videos = reconstruct_videos(
            U_1, U_3, G_flat, cfg['r_T'], cfg['H'], cfg['r_W']).float()
        videos_u8 = videos.clamp(0, 255).byte().numpy()  # (B, T, H, W)

        T = videos_u8.shape[1]
        idx = select_frames(T, args.n_frames)
        print(f'Selecting frame indices {idx} from T={T}')

        for i in range(bs):
            gi = produced + i
            frames = [videos_u8[i, t] for t in idx]
            path = os.path.join(
                args.outdir,
                f'preview_epoch{epoch:04d}_steps{args.sample_steps}_f{args.n_frames}_seed{args.seed}_{gi:03d}.png')
            save_row_png(frames, path)
            print(f'  saved {path}')

        produced += bs


if __name__ == '__main__':
    main()
