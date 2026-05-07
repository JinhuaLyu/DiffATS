"""
reconstruct_tucker.py -- Reconstruct videos from Tucker factors and save a
single (T, N, H, W) uint8 .pt tensor.

Two input modes:
    --shards_dir <dir>          Load all tucker_factors_shard_*.pt, use video_idx
                                to restore original ordering.
    --factors_pt  <file>         Load a single dict with U_1, U_3, G (as saved
                                 by generate.py in pt mode).
"""

import argparse
import glob
import os
import time

import torch


def reconstruct_batch(U_1, U_3, G):
    """Tucker reconstruction. Shapes:
        U_1: (B, T, r_T)
        U_3: (B, W, r_W)
        G  : (B, r_T, H, r_W)
    Returns (B, T, H, W) float video in raw range.
    """
    return torch.einsum('bahc,bta,bwc->bthw', G, U_1, U_3)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--shards_dir', type=str,
                   help='Dir with tucker_factors_shard_*.pt (U_1/U_3/G/video_idx)')
    g.add_argument('--factors_pt', type=str,
                   help='Single .pt with a dict {U_1, U_3, G}')
    p.add_argument('--out', required=True, type=str,
                   help='Output .pt path, will store (T, N, H, W) uint8 tensor.')
    p.add_argument('--order_by_idx', action='store_true',
                   help='(shards_dir mode) sort final output by video_idx')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    t0 = time.time()

    if args.shards_dir:
        paths = sorted(glob.glob(os.path.join(args.shards_dir,
                                              'tucker_factors_shard_*.pt')))
        print(f'Found {len(paths)} shards in {args.shards_dir}')
        all_video_idx = []
        all_videos = []
        for i, sp in enumerate(paths):
            s = torch.load(sp, map_location='cpu', weights_only=False)
            U1 = s['U_1'].to(device)          # (b, T, r_T)
            U3 = s['U_3'].to(device)          # (b, W, r_W)
            G  = s['G'].to(device)            # (b, r_T, H, r_W)
            vidx = s['video_idx']
            with torch.no_grad():
                v = reconstruct_batch(U1, U3, G)   # (b, T, H, W)
            v_u8 = v.clamp(0, 255).byte().cpu()
            all_videos.append(v_u8)
            all_video_idx.extend(vidx)
            if i % 5 == 0 or i == len(paths) - 1:
                print(f'  shard {i+1:2d}/{len(paths)}  cumN={sum(x.shape[0] for x in all_videos):5d}  '
                      f'elapsed={time.time()-t0:.1f}s', flush=True)

        videos = torch.cat(all_videos, dim=0)      # (N, T, H, W) uint8
        vidx_t = torch.tensor(all_video_idx, dtype=torch.int64)

        if args.order_by_idx:
            order = torch.argsort(vidx_t)
            videos = videos[order].contiguous()
            vidx_t = vidx_t[order].contiguous()
            print('Sorted by video_idx')

        videos_TNHW = videos.permute(1, 0, 2, 3).contiguous()   # (T, N, H, W) uint8
        payload = {
            'videos':       videos_TNHW,
            'video_idx':    vidx_t,
            'video_layout': '(T, N, H, W) uint8',
            'source':       'tucker_shards',
            'shards_dir':   os.path.abspath(args.shards_dir),
            'n_videos':     int(videos_TNHW.shape[1]),
        }

    else:
        f = torch.load(args.factors_pt, map_location='cpu', weights_only=False)
        U1, U3, G = f['U_1'], f['U_3'], f['G']
        print(f'Loaded {args.factors_pt}:  U1 {tuple(U1.shape)}  U3 {tuple(U3.shape)}  G {tuple(G.shape)}')
        N = U1.shape[0]
        bs = 500
        all_videos = []
        for i in range(0, N, bs):
            with torch.no_grad():
                v = reconstruct_batch(U1[i:i+bs].to(device),
                                      U3[i:i+bs].to(device),
                                      G[i:i+bs].to(device))
            all_videos.append(v.clamp(0, 255).byte().cpu())
            if (i // bs) % 5 == 0:
                print(f'  {i + all_videos[-1].shape[0]:5d}/{N}  '
                      f'elapsed={time.time()-t0:.1f}s', flush=True)
        videos = torch.cat(all_videos, dim=0)
        videos_TNHW = videos.permute(1, 0, 2, 3).contiguous()
        payload = {
            'videos':       videos_TNHW,
            'video_layout': '(T, N, H, W) uint8',
            'source':       'factors_pt',
            'factors_pt':   os.path.abspath(args.factors_pt),
            'n_videos':     int(videos_TNHW.shape[1]),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, args.out)
    print(f'Saved: {args.out}')
    print(f'  videos shape={tuple(videos_TNHW.shape)}  dtype={videos_TNHW.dtype}  '
          f'min={videos_TNHW.min().item()}  max={videos_TNHW.max().item()}  '
          f'mean={videos_TNHW.float().mean().item():.2f}')
    print(f'Total elapsed={time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
