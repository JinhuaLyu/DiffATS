from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dctn
from tqdm import tqdm


H_FRAME       = 128
W_FRAME       = 128
N_COND_TOKENS = 4
COND_LOW_FREQS = 26
LOW_FREQS_KEEP = N_COND_TOKENS * COND_LOW_FREQS  # 104
FIELDS         = ('ux', 'uy')
TAU_PERCENTILE = 99.0


def _zigzag_2d_full(H, W, n_keep):
    """Indices (in row-major flat order) of the n_keep lowest-frequency coefs
    of an HxW 2D DCT, sorted by the zigzag (i+j ascending) ordering."""
    coords = []
    for i in range(H):
        for j in range(W):
            coords.append((i + j, i, j))
    coords.sort()
    return np.array([i * W + j for _, i, j in coords[:n_keep]], dtype=np.int64)


def _encode_t0_shard(shard_obj, zz_keep):
    """Per-shard: for each clip's t=0 frame and each field, produce a
    (N_COND_TOKENS, COND_LOW_FREQS) tensor of raw 2D-DCT coefs."""
    n_clips = len(shard_obj)
    out = np.empty((n_clips * len(FIELDS), N_COND_TOKENS, COND_LOW_FREQS), dtype=np.float32)
    for i, sample in enumerate(shard_obj):
        for f_idx, field in enumerate(FIELDS):
            clip = sample[field]
            if isinstance(clip, torch.Tensor):
                clip = clip.numpy()
            t0 = clip[0].astype(np.float32, copy=False)         # (H, W)
            coefs2d = dctn(t0, type=2, norm='ortho', axes=(0, 1))  # (H, W)
            kept = coefs2d.reshape(-1)[zz_keep]                  # (LOW_FREQS_KEEP,)
            out[i * len(FIELDS) + f_idx] = kept.reshape(N_COND_TOKENS, COND_LOW_FREQS)
    return out


def _gather_raw(input_dir, prefix, zz_keep):
    shard_paths = sorted(glob.glob(os.path.join(input_dir, f'{prefix}*.pt')))
    if not shard_paths:
        raise FileNotFoundError(f'no {prefix}*.pt under {input_dir}')
    print(f'reading {len(shard_paths)} shards in {input_dir}')
    chunks = []
    for sp in tqdm(shard_paths, desc='shards'):
        obj = torch.load(sp, map_location='cpu', mmap=True, weights_only=False)
        chunks.append(_encode_t0_shard(obj, zz_keep))
        del obj
    arr = np.concatenate(chunks, axis=0)
    print(f'  raw t0-cond tensor: shape={arr.shape}, dtype={arr.dtype}, '
          f'size={arr.nbytes / (1024 ** 2):.1f} MB')
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir', default='/projects/p32954/jinhua_data/burgers_2d')
    ap.add_argument('--test-dir',  default='/projects/p32954/jinhua_data/burgers_2d_test')
    ap.add_argument('--out-dir',   default='/scratch/bkx8728/burgers_dctdiff_runs/burgers_dct_cache_b5')
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    out_train = os.path.join(args.out_dir, 'burgers_t0_cond_train.pt')
    out_test  = os.path.join(args.out_dir, 'burgers_t0_cond_test.pt')
    out_meta  = os.path.join(args.out_dir, 'burgers_t0_cond_meta.json')

    if args.skip_existing and os.path.isfile(out_train) and os.path.isfile(out_test) and os.path.isfile(out_meta):
        print('all t0-cond artifacts exist, skipping')
        return

    zz_keep = _zigzag_2d_full(H_FRAME, W_FRAME, LOW_FREQS_KEEP)
    print(f'kept {LOW_FREQS_KEEP} 2D-DCT coefs of the (128,128) t=0 frame '
          f'-> {N_COND_TOKENS} tokens of width {COND_LOW_FREQS}')
    print(f'compression ratio: {LOW_FREQS_KEEP / (H_FRAME * W_FRAME) * 100:.2f}% of t=0 pixels')

    # Train set first -- compute Y_bound_t0 from train statistics.
    train_raw = _gather_raw(args.train_dir, 'shard_', zz_keep)
    y_bound_t0 = float(np.percentile(np.abs(train_raw), TAU_PERCENTILE))
    print(f'Y_bound_t0 (p{TAU_PERCENTILE} of |coefs| over train): {y_bound_t0:.6f}')

    test_raw = _gather_raw(args.test_dir, 'test_shard_', zz_keep)

    # Save normalised tensors so the dataset can return them as-is (no further divide).
    train_norm = (train_raw / y_bound_t0).astype(np.float32, copy=False)
    test_norm  = (test_raw  / y_bound_t0).astype(np.float32, copy=False)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(torch.from_numpy(train_norm), out_train)
    torch.save(torch.from_numpy(test_norm),  out_test)
    Path(out_meta).write_text(json.dumps({
        'Y_bound_t0': y_bound_t0,
        'n_cond_tokens': N_COND_TOKENS,
        'cond_low_freqs': COND_LOW_FREQS,
        'low_freqs_keep': LOW_FREQS_KEEP,
        'tau_percentile': TAU_PERCENTILE,
        'fields': list(FIELDS),
    }, indent=2))
    print(f'saved {out_train} ({train_norm.nbytes/1024**2:.1f} MB)')
    print(f'saved {out_test}  ({test_norm.nbytes/1024**2:.1f} MB)')
    print(f'saved {out_meta}')


if __name__ == '__main__':
    main()
