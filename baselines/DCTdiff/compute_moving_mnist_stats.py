"""Compute 3D-DCT statistics from the Moving MNIST .pt file.

Each clip (T,H,W) is split into 3D blocks of shape (BLOCK_T, BLOCK_HW, BLOCK_HW),
3D DCT is applied per block, coefficients are reordered by 3D zigzag (ascending
i+j+k), and Y_bound + per-position std are computed.

Saves to mm_stats_3d.json so dct_train.py can load them automatically.

Usage:
    python dct_3d_statis.py
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from DCT_utils import (
    split_clip_into_blocks_3d, dct3_block, zigzag_order_3d,
)

DATA_PATH   = '/home/bkx8728/Tensor_factor/moving_mnist/moving_mnist_20k_2slow.pt'
STATS_PATH  = Path(__file__).parent / 'mm_stats_3d.json'

BLOCK_T   = 4
BLOCK_HW  = 8
LOW_FREQS = 64    # number of zigzag coefficients to keep per 3D block (320*64 = 20480)
N_CLIPS   = 4_000  # number of clips to sample for statistics
TAU       = 99.0


def main():
    print(f'Loading {DATA_PATH} ...')
    data = torch.load(DATA_PATH, map_location='cpu', weights_only=False)
    if data.dim() == 5:
        data = data.squeeze(2)
    assert data.dim() == 4, f"expected 4D tensor, got {tuple(data.shape)}"
    # Normalize axis order to (N, T, H, W). The .pt is stored as (T, N, H, W)
    # for Moving MNIST, so transpose if the first axis is smaller than the second.
    if data.shape[0] < data.shape[1]:
        print(f'  transposing axes 0,1 (was {tuple(data.shape)})')
        data = data.permute(1, 0, 2, 3).contiguous()
    N, T, H, W = data.shape
    print(f'  shape: N={N}, T={T}, H={H}, W={W}')
    print(f'  value range: [{data.min():.4f}, {data.max():.4f}]')

    assert T % BLOCK_T == 0 and H % BLOCK_HW == 0 and W % BLOCK_HW == 0, (
        f'clip ({T},{H},{W}) not divisible by ({BLOCK_T},{BLOCK_HW},{BLOCK_HW})'
    )

    zz = zigzag_order_3d(BLOCK_T, BLOCK_HW, BLOCK_HW)
    coefs_per_block = BLOCK_T * BLOCK_HW * BLOCK_HW
    num_blocks_per_clip = (T // BLOCK_T) * (H // BLOCK_HW) * (W // BLOCK_HW)

    rng = np.random.default_rng(0)
    n_use = min(N_CLIPS, N)
    seq_idxs = rng.choice(N, size=n_use, replace=False)
    print(f'  sampling {n_use} clips for stats')

    # accumulate sum, sum-of-squares, count for std; collect first LOW_FREQS coefs
    # as a flat list across blocks for the percentile.
    sum_sq = np.zeros(coefs_per_block, dtype=np.float64)
    sum_   = np.zeros(coefs_per_block, dtype=np.float64)
    count  = 0
    abs_low = []

    for i in tqdm(seq_idxs, desc='computing 3D DCT stats'):
        clip = data[int(i)].numpy().astype(np.float32)  # (T, H, W)
        blocks = split_clip_into_blocks_3d(clip, BLOCK_T, BLOCK_HW, BLOCK_HW)  # (nB, bT, bH, bW)
        for blk in blocks:
            c = dct3_block(blk).reshape(coefs_per_block)[zz]
            sum_   += c
            sum_sq += c * c
        count += blocks.shape[0]

        # gather the first LOW_FREQS zigzag coefficients across all blocks
        clip_coefs = np.empty((blocks.shape[0], LOW_FREQS), dtype=np.float32)
        for j, blk in enumerate(blocks):
            clip_coefs[j] = dct3_block(blk).reshape(coefs_per_block)[zz][:LOW_FREQS]
        abs_low.append(np.abs(clip_coefs))

    abs_low = np.concatenate(abs_low, axis=0)  # (n_use * num_blocks_per_clip, LOW_FREQS)
    y_bound = float(np.percentile(abs_low, TAU))

    mean_ = sum_ / count
    var_  = np.maximum(sum_sq / count - mean_ * mean_, 0.0)
    std_  = np.sqrt(var_)
    vor_std = [round(float(v), 6) for v in std_.tolist()]

    stats = {
        'Y_bound': y_bound,
        'vor_std': vor_std,
        'block_T': BLOCK_T,
        'block_HW': BLOCK_HW,
        'low_freqs': LOW_FREQS,
        'num_blocks_per_clip': int(num_blocks_per_clip),
        'n_clips_used': int(n_use),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2))

    print(f'\nY_bound (99th pct of |first {LOW_FREQS}|): {y_bound:.6f}')
    print(f'vor_std (zigzag order, first {LOW_FREQS}): {vor_std[:LOW_FREQS]}')
    print(f'\nSaved → {STATS_PATH}')


if __name__ == '__main__':
    main()
