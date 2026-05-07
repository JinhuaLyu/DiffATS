"""
reconstruct_gen.py — Reconstruct videos from saved generated Tucker factors.

For one seed of each experiment, load epoch00200_seed{seed}.pt, compute
  video = einsum(U_T/U1, U_Y/U3, G)
for every sample, and save a single .pt with:
  {'videos': (N, 200, 128, 128) float32, 'sample_idx': (N,), ...}
"""
import argparse
import os
import sys
import time

import numpy as np
import torch


def reconstruct_karman(U_T, U_Y, G):
    # U_T:(T,rT)  U_Y:(W,rY)  G:(rT,X,rY)
    temp = np.einsum('txk,wk->txw', G, U_Y, optimize=True)
    return np.einsum('ti,ixw->txw', U_T, temp, optimize=True)


def reconstruct_burgers(U1, U3, G):
    # U1:(T,rT)  U3:(W,rW)  G:(rT,H,rW)
    temp = np.einsum('thk,wk->thw', G, U3, optimize=True)
    return np.einsum('ti,ihw->thw', U1, temp, optimize=True)


def process(exp, in_path, out_path):
    d = torch.load(in_path, map_location='cpu', weights_only=False)
    if exp == 'karman':
        A = d['U_T'].numpy().astype(np.float32)
        B = d['U_Y'].numpy().astype(np.float32)
        G = d['G'  ].numpy().astype(np.float32)
        recon_fn = reconstruct_karman
    else:
        A = d['U1' ].numpy().astype(np.float32)
        B = d['U3' ].numpy().astype(np.float32)
        G = d['G'  ].numpy().astype(np.float32)
        recon_fn = reconstruct_burgers
    N = A.shape[0]
    print(f'[{exp}] N={N}  reconstructing...', flush=True)

    videos = np.empty((N, 200, 128, 128), dtype=np.float32)
    t0 = time.time()
    for i in range(N):
        videos[i] = recon_fn(A[i], B[i], G[i])
        if (i + 1) % 100 == 0 or (i + 1) == N:
            print(f'  [{exp}] {i+1}/{N}  elapsed={time.time()-t0:.1f}s',
                  flush=True)

    out = {
        'videos'       : torch.from_numpy(videos),
        'sample_idx'   : d['sample_idx'],
        'seed'         : d.get('seed'),
        'epoch'        : d.get('epoch'),
        'step'         : d.get('step'),
        'ckpt_path'    : d.get('ckpt_path'),
        'sample_steps' : d.get('sample_steps'),
        'source_file'  : in_path,
    }
    if exp == 'burgers':
        out['nu'] = d['nu']; out['cd'] = d['cd']
    else:
        out['niu'] = d['niu']; out['Re'] = d['Re']
        out['cx']  = d['cx'];  out['cy'] = d['cy']; out['r'] = d['r']

    torch.save(out, out_path)
    size_gb = os.path.getsize(out_path) / 1024**3
    print(f'[{exp}] saved {out_path}  ({size_gb:.2f} GB)  '
          f'total={time.time()-t0:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['karman', 'burgers'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--dir', type=str, default=None,
                        help='Override default output_dir.')
    args = parser.parse_args()

    if args.dir is None:
        sub = 'karman_vortex_2d' if args.exp == 'karman' else 'burgers_2d'
        args.dir = f'${DATA_ROOT}/our_method_generation/{sub}'

    tag     = f'epoch{args.epoch:05d}'
    in_path = os.path.join(args.dir, f'{tag}_seed{args.seed}.pt')
    out_path= os.path.join(args.dir, f'{tag}_seed{args.seed}_videos.pt')
    process(args.exp, in_path, out_path)


if __name__ == '__main__':
    main()
