"""
generate.py -- Sample videos from a trained Tucker3PartsDiT checkpoint.

Modes:
    gif : save each video as an individual .gif  (for visual preview)
    pt  : accumulate all videos into a single (T, N, H, W) uint8 tensor

Usage:
    # preview 10 videos as GIFs
    python generate.py \\
        --ckpt .../epoch2000.pt \\
        --n_videos 10 --batch_size 10 --mode gif \\
        --outdir ./data

    # full batch of 10000, saved as single .pt
    python generate.py \\
        --ckpt .../epoch2000.pt \\
        --n_videos 10000 --batch_size 64 --mode pt \\
        --outdir /anvil/projects/x-eng260004/factor_diffusion/our_method_generation/moving_mnist
"""

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion import create_diffusion
from model import build_tucker_3parts_dit
from train import unpack_and_denormalize, reconstruct_videos


FLAT_SPLIT  = [300, 1280, 19200]          # T*r_T, W*r_W, r_T*H*r_W
FLAT_SHAPES = [(20, 15), (64, 20), (960, 20)]


def save_gif(frames_u8, path, duration_ms=80):
    if isinstance(frames_u8, torch.Tensor):
        frames_u8 = frames_u8.cpu().numpy()
    imgs = [Image.fromarray(f, mode='L') for f in frames_u8]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=duration_ms, loop=0)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',         required=True, type=str)
    p.add_argument('--n_videos',     required=True, type=int)
    p.add_argument('--batch_size',   type=int, default=64)
    p.add_argument('--mode',         choices=['gif', 'pt'], required=True)
    p.add_argument('--outdir',       required=True, type=str)
    p.add_argument('--seed',         type=int, default=0)
    p.add_argument('--sample_steps', type=int, default=None,
                   help='Override cfg[sample_steps]')
    p.add_argument('--scale',        type=float, default=1.0,
                   help='Multiply reconstructed video by this factor, then clip to [0,255].')
    p.add_argument('--low_cut',      type=float, default=0.0,
                   help='Pixels below this fraction of 255 are zeroed, remainder is '
                        'linearly stretched back to [0, 255] before --scale is applied.')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg       = ckpt['cfg']
    raw_stats = ckpt['stats']
    epoch     = ckpt['epoch']
    print(f'Checkpoint: epoch={epoch}  step={ckpt["step"]}')
    print(f'  cfg: T={cfg["T"]} H={cfg["H"]} W={cfg["W"]} '
          f'r_T={cfg["r_T"]} r_W={cfg["r_W"]} '
          f'sample_steps={cfg["sample_steps"]} noise_schedule={cfg["noise_schedule"]}')

    wrapper = build_tucker_3parts_dit(cfg).to(device)
    wrapper.load_state_dict(ckpt['ema'])
    wrapper.eval()
    n_params = sum(q.numel() for q in wrapper.parameters())
    print(f'Loaded EMA weights. params={n_params:,}')

    sample_steps = args.sample_steps if args.sample_steps is not None else cfg['sample_steps']
    diffusion_sample = create_diffusion(
        timestep_respacing=str(sample_steps),
        noise_schedule=cfg['noise_schedule'],
        learn_sigma=False,
        diffusion_steps=cfg['diffusion_steps'],
    )
    print(f'Diffusion sampler: sample_steps={sample_steps}  '
          f'diffusion_steps={cfg["diffusion_steps"]}')

    stats = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw_stats.items()}

    os.makedirs(args.outdir, exist_ok=True)

    flat_dim = sum(FLAT_SPLIT)
    buf_U1, buf_U3, buf_G, buf_videos = [], [], [], []
    produced = 0

    while produced < args.n_videos:
        bs = min(args.batch_size, args.n_videos - produced)
        x_flat = torch.randn(bs, flat_dim, device=device)
        samples = diffusion_sample.p_sample_loop(
            wrapper, x_flat.shape, noise=x_flat,
            clip_denoised=False, model_kwargs={},
            device=device, progress=False,
        )
        U_1, U_3, G_flat = unpack_and_denormalize(
            samples.float().cpu(), stats, FLAT_SPLIT, FLAT_SHAPES, 'cpu')
        videos = reconstruct_videos(U_1, U_3, G_flat, cfg['r_T'], cfg['H'], cfg['r_W']).float()
        if args.low_cut > 0:
            thr = 255.0 * args.low_cut
            videos = ((videos - thr).clamp_min(0) / max(1 - args.low_cut, 1e-6))
        videos_u8 = (videos * args.scale).clamp(0, 255).byte()

        tag = ''
        if args.low_cut > 0:
            tag += f'_lc{args.low_cut:g}'
        if args.scale != 1.0:
            tag += f'_x{args.scale:g}'

        if args.mode == 'gif':
            for i in range(bs):
                gi = produced + i
                path = os.path.join(args.outdir, f'gen_epoch{epoch:04d}{tag}_{gi:03d}.gif')
                save_gif(videos_u8[i], path)
            print(f'  GIFs {produced:04d}..{produced + bs - 1:04d} saved')
        else:
            buf_U1.append(U_1.contiguous())
            buf_U3.append(U_3.contiguous())
            buf_G.append(G_flat.reshape(bs, cfg['r_T'], cfg['H'], cfg['r_W']).contiguous())
            buf_videos.append(videos_u8)
            print(f'  {produced + bs}/{args.n_videos}')

        produced += bs

    if args.mode == 'pt':
        U1_all = torch.cat(buf_U1, dim=0)                           # (N, T, r_T)
        U3_all = torch.cat(buf_U3, dim=0)                           # (N, W, r_W)
        G_all  = torch.cat(buf_G,  dim=0)                           # (N, r_T, H, r_W)
        videos_NTHW = torch.cat(buf_videos, dim=0)                   # (N, T, H, W) uint8
        videos_TNHW = videos_NTHW.permute(1, 0, 2, 3).contiguous()   # (T, N, H, W) uint8

        video_tag = ''
        if args.low_cut > 0:
            video_tag += f'_lc{args.low_cut:g}'
        if args.scale != 1.0:
            video_tag += f'_x{args.scale:g}'

        common_meta = {
            'n_videos':       int(U1_all.shape[0]),
            'epoch':          int(epoch),
            'step':           int(ckpt['step']),
            'ckpt_path':      os.path.abspath(args.ckpt),
            'sample_steps':   int(sample_steps),
            'noise_schedule': str(cfg['noise_schedule']),
            'T_diffusion':    int(cfg['diffusion_steps']),
            'seed':           int(args.seed),
        }

        # -- File 1: raw Tucker factors (unaffected by --scale/--low_cut) --
        factors_path = os.path.join(
            args.outdir, f'moving_mnist_gen_epoch{epoch:04d}_factors.pt')
        factors_payload = {
            'U_1':           U1_all,
            'U_3':           U3_all,
            'G':             G_all,
            'factor_layout': {'U_1': '(N, T, r_T)',
                              'U_3': '(N, W, r_W)',
                              'G':   '(N, r_T, H, r_W)'},
            **common_meta,
        }
        torch.save(factors_payload, factors_path)
        print(f'Saved: {factors_path}')
        print(f'  U_1 shape={tuple(U1_all.shape)}  dtype={U1_all.dtype}')
        print(f'  U_3 shape={tuple(U3_all.shape)}  dtype={U3_all.dtype}')
        print(f'  G   shape={tuple(G_all.shape)}  dtype={G_all.dtype}')

        # -- File 2: reconstructed + brightness-enhanced videos --
        videos_path = os.path.join(
            args.outdir, f'moving_mnist_gen_epoch{epoch:04d}{video_tag}_videos.pt')
        videos_payload = {
            'videos':       videos_TNHW,
            'video_layout': '(T, N, H, W) uint8',
            'low_cut':      float(args.low_cut),
            'scale':        float(args.scale),
            **common_meta,
        }
        torch.save(videos_payload, videos_path)
        print(f'Saved: {videos_path}')
        print(f'  videos shape={tuple(videos_TNHW.shape)}  dtype={videos_TNHW.dtype}')


if __name__ == '__main__':
    main()
