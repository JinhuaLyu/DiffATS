import json
import glob
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from DCT_utils import split_clip_into_blocks_3d, dct3_block, zigzag_order_3d


TRAIN_DIR = '${DATA_ROOT}/bkx8728/karman_vortex_2d'
STATS_PATH = Path(__file__).parent / 'karman_stats_3d.json'

T_USE     = 200   
BLOCK_T   = 50
BLOCK_H   = 16
BLOCK_W   = 16
LOW_FREQS = 98         
N_SHARDS  = 10        
TAU       = 99.0


def main():
    shard_paths = sorted(glob.glob(os.path.join(TRAIN_DIR, 'shard_*.pt')))
    if not shard_paths:
        raise FileNotFoundError(f'no shard_*.pt under {TRAIN_DIR}')
    print(f'found {len(shard_paths)} training shards in {TRAIN_DIR}')
    shard_paths = shard_paths[:N_SHARDS]
    print(f'  using only the first {len(shard_paths)} shards for stats')

    # Pull first shard to confirm shape
    first = torch.load(shard_paths[0], map_location='cpu', weights_only=False)
    T0, H0, W0 = first[0]['vor'].shape
    print(f'  shard[0][0].vor: ({T0}, {H0}, {W0})  (will crop T -> {T_USE})')
    assert T_USE <= T0, f'T_USE={T_USE} > T={T0}'
    assert T_USE % BLOCK_T == 0 and H0 % BLOCK_H == 0 and W0 % BLOCK_W == 0, (
        f'({T_USE},{H0},{W0}) not divisible by ({BLOCK_T},{BLOCK_H},{BLOCK_W})'
    )

    zz = zigzag_order_3d(BLOCK_T, BLOCK_H, BLOCK_W)
    coefs_per_block = BLOCK_T * BLOCK_H * BLOCK_W
    num_blocks_per_clip = (T_USE // BLOCK_T) * (H0 // BLOCK_H) * (W0 // BLOCK_W)
    print(f'  num_blocks_per_clip = {num_blocks_per_clip}, total kept = '
          f'{num_blocks_per_clip * LOW_FREQS}')

    clips_per_shard = len(first)
    print(f'  clips/shard = {clips_per_shard}, total clips for stats = '
          f'{clips_per_shard * len(shard_paths)}')

    sum_   = np.zeros(coefs_per_block, dtype=np.float64)
    sum_sq = np.zeros(coefs_per_block, dtype=np.float64)
    count  = 0
    abs_low_chunks = []

    from scipy.fft import dctn

    for s_idx, sp in enumerate(shard_paths):
        if s_idx == 0:
            shard_obj = first
        else:
            shard_obj = torch.load(sp, map_location='cpu', weights_only=False)
        for local in tqdm(range(len(shard_obj)),
                          desc=f'shard {s_idx + 1}/{len(shard_paths)}'):
            clip = shard_obj[local]['vor']
            if isinstance(clip, torch.Tensor):
                clip = clip.numpy()
            clip = clip.astype(np.float32, copy=False)[:T_USE]

            blocks = split_clip_into_blocks_3d(clip, BLOCK_T, BLOCK_H, BLOCK_W)
            # Batched DCT: one call for all 80 blocks.
            dct_blocks = dctn(blocks, type=2, norm='ortho', axes=(1, 2, 3))
            coefs = dct_blocks.reshape(blocks.shape[0], coefs_per_block)[:, zz]
            sum_   += coefs.sum(axis=0)
            sum_sq += (coefs ** 2).sum(axis=0)
            count  += blocks.shape[0]
            abs_low_chunks.append(np.abs(coefs[:, :LOW_FREQS]))

    abs_low = np.concatenate(abs_low_chunks, axis=0)
    y_bound = float(np.percentile(abs_low, TAU))

    mean_ = sum_ / count
    var_  = np.maximum(sum_sq / count - mean_ * mean_, 0.0)
    std_  = np.sqrt(var_)
    vor_std = [round(float(v), 8) for v in std_.tolist()]

    stats = {
        'Y_bound': y_bound,
        'vor_std': vor_std,
        'T_use': T_USE,
        'block_T': BLOCK_T,
        'block_H': BLOCK_H,
        'block_W': BLOCK_W,
        'low_freqs': LOW_FREQS,
        'num_blocks_per_clip': int(num_blocks_per_clip),
        'n_clips_used': int(count // num_blocks_per_clip),
        'total_kept_coefs': int(num_blocks_per_clip * LOW_FREQS),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2))

    print(f'\nY_bound (p{TAU} of |first {LOW_FREQS}|): {y_bound:.6f}')
    print(f'vor_std[:8]: {vor_std[:8]}')
    print(f'\nSaved -> {STATS_PATH}')


if __name__ == '__main__':
    main()
