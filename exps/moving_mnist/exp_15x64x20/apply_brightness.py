"""
apply_brightness.py -- Apply (low_cut, scale) brightening to a saved video .pt.

Same formula as generate.py:
    x' = clamp((x - 255*low_cut) / (1 - low_cut), 0, ...)
    y  = clamp(x' * scale, 0, 255)  -> uint8
"""

import argparse
import os
import time

import torch


def brighten(videos_u8, low_cut: float, scale: float):
    x = videos_u8.float()
    if low_cut > 0:
        thr = 255.0 * low_cut
        x = (x - thr).clamp_min(0) / max(1 - low_cut, 1e-6)
    y = (x * scale).clamp(0, 255).byte()
    return y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inp',      required=True, type=str)
    p.add_argument('--out',      required=True, type=str)
    p.add_argument('--key',      type=str, default='videos',
                   help='Dict key if input is a dict')
    p.add_argument('--low_cut',  type=float, default=0.1)
    p.add_argument('--scale',    type=float, default=2.5)
    args = p.parse_args()

    t0 = time.time()
    obj = torch.load(args.inp, map_location='cpu', weights_only=False)
    if isinstance(obj, dict):
        vids = obj[args.key]
    else:
        vids = obj
    assert vids.dtype == torch.uint8
    print(f'Loaded {args.inp}  shape={tuple(vids.shape)}')

    vids_b = brighten(vids, args.low_cut, args.scale)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # Keep same container layout: if input was dict, preserve metadata; else save bare tensor
    if isinstance(obj, dict):
        new_obj = {**obj, args.key: vids_b,
                   'low_cut': float(args.low_cut),
                   'scale':   float(args.scale)}
        torch.save(new_obj, args.out)
    else:
        torch.save(vids_b, args.out)

    print(f'Saved {args.out}  shape={tuple(vids_b.shape)}  dtype={vids_b.dtype}  '
          f'min={vids_b.min().item()}  max={vids_b.max().item()}  '
          f'mean={vids_b.float().mean().item():.2f}')
    print(f'Elapsed: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
