"""Precompute the truncated 3D-DCT representation of every Karman clip.

For each clip:
  1. crop to T_USE frames
  2. split into (block_T, block_H, block_W) blocks
  3. apply orthonormal 3D DCT (batched per shard)
  4. reorder by zigzag (i+j+k ascending)
  5. keep the first LOW_FREQS coefficients

Output: a single .pt file with a float32 tensor of shape (N_clips, num_blocks, LOW_FREQS).

Storage cost (float32): N_clips * num_blocks * LOW_FREQS * 4 bytes
    train: 10000 * 80 * 313 * 4  ~=  1.0 GB
    test:    500 * 80 * 313 * 4  ~=  50 MB

Usage:
    python precompute_karman_dct.py
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dctn
from tqdm import tqdm

from DCT_utils import split_clip_into_blocks_3d, zigzag_order_3d, zigzag_order_2d


# Parameters must match karman_3d_statis.py / configs/karman_uvit_3d.py
T_USE     = 200
BLOCK_T   = 50
BLOCK_H   = 16
BLOCK_W   = 16
LOW_FREQS = 98
COND_DIM  = 1024     # 2D-DCT zigzag low-freq coefs of t=0 frame, used as conditioning


def encode_shard(shard_obj, zz, num_blocks, coefs_per_block, zz_2d):
    """Encode shard -> (tokens, t0_cond) tensors.

    tokens:  (n_clips, num_blocks, LOW_FREQS)
    t0_cond: (n_clips, COND_DIM) — 2D-DCT zigzag low-freq of frame 0
    """
    n = len(shard_obj)
    out_tok  = np.empty((n, num_blocks, LOW_FREQS), dtype=np.float32)
    out_cond = np.empty((n, COND_DIM), dtype=np.float32)
    for i, sample in enumerate(shard_obj):
        clip = sample['vor']
        if isinstance(clip, torch.Tensor):
            clip = clip.numpy()
        clip = clip.astype(np.float32, copy=False)[:T_USE]

        blocks = split_clip_into_blocks_3d(clip, BLOCK_T, BLOCK_H, BLOCK_W)
        dct_blocks = dctn(blocks, type=2, norm='ortho', axes=(1, 2, 3))
        coefs = dct_blocks.reshape(num_blocks, coefs_per_block)[:, zz][:, :LOW_FREQS]
        out_tok[i] = coefs

        # t=0 conditioning: 2D-DCT of frame 0
        t0 = clip[0]
        t0_dct = dctn(t0, type=2, norm='ortho').reshape(-1)
        out_cond[i] = t0_dct[zz_2d][:COND_DIM]
    return out_tok, out_cond


def precompute(input_dir, prefix, out_tokens_path, out_cond_path):
    shard_paths = sorted(glob.glob(os.path.join(input_dir, f'{prefix}*.pt')))
    if not shard_paths:
        raise FileNotFoundError(f'no {prefix}*.pt under {input_dir}')
    print(f'precomputing from {len(shard_paths)} shards in {input_dir}')
    print(f'  tokens -> {out_tokens_path}')
    print(f'  cond   -> {out_cond_path}')

    zz = zigzag_order_3d(BLOCK_T, BLOCK_H, BLOCK_W)
    zz_2d = zigzag_order_2d(128, 128)
    coefs_per_block = BLOCK_T * BLOCK_H * BLOCK_W
    num_blocks = (T_USE // BLOCK_T) * (128 // BLOCK_H) * (128 // BLOCK_W)
    print(f'  num_blocks/clip = {num_blocks}, low_freqs = {LOW_FREQS}, '
          f'kept/clip = {num_blocks * LOW_FREQS},  cond_dim = {COND_DIM}')

    tok_chunks, cond_chunks = [], []
    for sp in tqdm(shard_paths, desc='shards'):
        obj = torch.load(sp, map_location='cpu', weights_only=False)
        tok, cond = encode_shard(obj, zz, num_blocks, coefs_per_block, zz_2d)
        tok_chunks.append(tok)
        cond_chunks.append(cond)
        del obj

    arr_tok  = np.concatenate(tok_chunks, axis=0)
    arr_cond = np.concatenate(cond_chunks, axis=0)
    print(f'tokens: shape={arr_tok.shape}, size={arr_tok.nbytes / (1024 ** 3):.3f} GB')
    print(f'cond:   shape={arr_cond.shape}, size={arr_cond.nbytes / (1024 ** 2):.1f} MB')

    out_dir = os.path.dirname(out_tokens_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(torch.from_numpy(arr_tok),  out_tokens_path)
    torch.save(torch.from_numpy(arr_cond), out_cond_path)
    print(f'  saved tokens -> {out_tokens_path}')
    print(f'  saved cond   -> {out_cond_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='${DATA_ROOT}/bkx8728/karman_vortex_2d')
    parser.add_argument('--test-dir',  default='${DATA_ROOT}/bkx8728/karman_vortex_2d/test_data')
    parser.add_argument('--out-dir',   default='${DATA_ROOT}/bkx8728/karman_vortex_2d/dct_cache',
                        help='output directory for the cached .pt files (~1.05 GB tokens + ~50 MB cond)')
    parser.add_argument('--skip-test', action='store_true')
    args = parser.parse_args()

    precompute(
        args.train_dir, 'shard_',
        os.path.join(args.out_dir, 'karman_dct_train.pt'),
        os.path.join(args.out_dir, 'karman_t0_cond_train.pt'),
    )

    if not args.skip_test and args.test_dir and os.path.isdir(args.test_dir):
        precompute(
            args.test_dir, 'test_shard_',
            os.path.join(args.out_dir, 'karman_dct_test.pt'),
            os.path.join(args.out_dir, 'karman_t0_cond_test.pt'),
        )


if __name__ == '__main__':
    main()
