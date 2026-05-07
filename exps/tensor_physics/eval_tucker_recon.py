"""
eval_tucker_recon.py — Sample random rows from Tucker factors (train+test union)
and compute mean relative reconstruction error vs raw data.

For each sampled row we compute two relative errors:
  1. Tucker trajectory recon : ||einsum(U, C, ...) - raw[1:]||_F / ||raw[1:]||_F
  2. IC SVD recon            : ||U_ic @ Vh_ic - raw[0]||_F / ||raw[0]||_F

Usage:
  python eval_tucker_recon.py --exp burgers
  python eval_tucker_recon.py --exp karman
  python eval_tucker_recon.py --exp both      # run both sequentially
"""

import argparse
import os
import random
from collections import defaultdict

# BLAS threading is catastrophic for tiny matrices (moving_mnist videos are
# 20×64×64; multi-threaded OpenBLAS spends all its time in thread overhead).
# Must be set before numpy is imported.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'MKL_NUM_THREADS', 'BLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------

CONFIGS = {
    'burgers': {
        'tucker_train_dir': '/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/burgers_2d/tucker_burgers_rT5_rH20_rW20',
        'tucker_test_dir':  '/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/burgers_2d/tucker_burgers_rT5_rH20_rW20/test_data',
        'raw_train_dir':    '/anvil/projects/x-eng260004/factor_diffusion/original_data/burgers_2d',
        'raw_test_dir':     '/anvil/projects/x-eng260004/factor_diffusion/original_data/burgers_2d/test_data',
        'raw_prefix_train': 'shard',           # → shard_00000.pt
        'raw_prefix_test':  'test_shard',      # → test_shard_00000.pt
        'raw_digits':       5,
        'raw_samples_per_shard': 100,          # physical samples per raw shard
        'has_two_fields':   True,              # ux (even sample_idx) + uy (odd)
    },
    'karman': {
        'tucker_train_dir': '/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/karman_vortex_2d/tucker_karman_rT10_rX128_rY30',
        'tucker_test_dir':  '/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/karman_vortex_2d/tucker_karman_rT10_rX128_rY30/test_data',
        'raw_train_dir':    '/anvil/projects/x-eng260004/factor_diffusion/original_data/karman_vortex_2d',
        'raw_test_dir':     '/anvil/projects/x-eng260004/factor_diffusion/original_data/karman_vortex_2d/test_data',
        'raw_prefix_train': 'shard',
        'raw_prefix_test':  'test_shard',
        'raw_digits':       3,
        'raw_samples_per_shard': 50,
        'has_two_fields':   False,             # single vor field
    },
}


# ---------------------------------------------------------------------------
# Shard enumeration
# ---------------------------------------------------------------------------

def list_shards(tucker_dir, source_label):
    """Return list of (source, tucker_shard_path) from manifest.txt."""
    manifest = os.path.join(tucker_dir, 'manifest.txt')
    with open(manifest) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return [(source_label, os.path.join(tucker_dir, n)) for n in names]


# ---------------------------------------------------------------------------
# Raw shard loader (map sample_idx → raw sample)
# ---------------------------------------------------------------------------

def raw_shard_path(cfg, source, phys_idx):
    per = cfg['raw_samples_per_shard']
    digits = cfg['raw_digits']
    shard_idx = phys_idx // per
    local = phys_idx % per
    if source == 'train':
        prefix = cfg['raw_prefix_train']
        d = cfg['raw_train_dir']
    else:
        prefix = cfg['raw_prefix_test']
        d = cfg['raw_test_dir']
    path = os.path.join(d, f'{prefix}_{shard_idx:0{digits}d}.pt')
    return path, local


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_burgers(tucker_row: dict):
    """burgers row → (video_recon, U_ic, Vh_ic)."""
    U_1 = tucker_row['U_1'].numpy().astype(np.float64)   # (200, r_T)
    U_2 = tucker_row['U_2'].numpy().astype(np.float64)   # (128, r_H)
    U_3 = tucker_row['U_3'].numpy().astype(np.float64)   # (128, r_W)
    C   = tucker_row['C'].numpy().astype(np.float64)     # (r_T, r_H, r_W)
    U_ic  = tucker_row['U_ic'].numpy().astype(np.float64)
    Vh_ic = tucker_row['Vh_ic'].numpy().astype(np.float64)
    video = np.einsum('ti,hj,wk,ijk->thw', U_1, U_2, U_3, C, optimize=True)
    return video, U_ic, Vh_ic


def reconstruct_karman(tucker_row: dict):
    """karman row → (video_recon, U_ic, Vh_ic)."""
    U_T = tucker_row['U_T'].numpy().astype(np.float64)   # (200, r_T)
    U_X = tucker_row['U_X'].numpy().astype(np.float64)   # (128, r_X=128)
    U_Y = tucker_row['U_Y'].numpy().astype(np.float64)   # (128, r_Y)
    C   = tucker_row['C'].numpy().astype(np.float64)     # (r_T, r_X, r_Y)
    U_ic  = tucker_row['U_ic'].numpy().astype(np.float64)
    Vh_ic = tucker_row['Vh_ic'].numpy().astype(np.float64)
    video = np.einsum('ti,xj,wk,ijk->txw', U_T, U_X, U_Y, C, optimize=True)
    return video, U_ic, Vh_ic


def extract_row(tucker_shard: dict, row_idx: int, exp: str) -> dict:
    """Slice one row out of a loaded tucker shard."""
    if exp == 'burgers':
        keys = ['U_1', 'U_2', 'U_3', 'C', 'U_ic', 'Vh_ic']
    else:
        keys = ['U_T', 'U_X', 'U_Y', 'C', 'U_ic', 'Vh_ic']
    out = {k: tucker_shard[k][row_idx] for k in keys}
    # metadata
    out['sample_idx'] = int(tucker_shard['sample_idx'][row_idx])
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def evaluate(exp: str, n_shards: int = 10, rows_per_shard: int = 10, seed: int = 0):
    cfg = CONFIGS[exp]
    rng = random.Random(seed)

    shards = (list_shards(cfg['tucker_train_dir'], 'train')
              + list_shards(cfg['tucker_test_dir'],  'test'))

    chosen = rng.sample(shards, min(n_shards, len(shards)))
    print(f'[{exp}] drew {len(chosen)} tucker shards '
          f'({sum(1 for s,_ in chosen if s=="train")} train / '
          f'{sum(1 for s,_ in chosen if s=="test")} test)')

    traj_errs, ic_errs = [], []

    recon_fn = reconstruct_burgers if exp == 'burgers' else reconstruct_karman

    for source, tpath in chosen:
        tucker = torch.load(tpath, map_location='cpu', weights_only=False)
        B = tucker['sample_idx'] if isinstance(tucker['sample_idx'], list) \
            else list(tucker['sample_idx'])
        n_rows = len(B)
        row_idxs = rng.sample(range(n_rows), min(rows_per_shard, n_rows))

        # Group rows by their raw shard to minimize raw I/O
        groups = defaultdict(list)
        for ri in row_idxs:
            sid = int(tucker['sample_idx'][ri])
            if cfg['has_two_fields']:
                phys_idx = sid // 2
                field = 'ux' if (sid % 2 == 0) else 'uy'
            else:
                phys_idx = sid
                field = 'vor'
            path, local = raw_shard_path(cfg, source, phys_idx)
            groups[path].append((ri, local, field, sid))

        for raw_path, infos in groups.items():
            raw = torch.load(raw_path, map_location='cpu', weights_only=False)
            for ri, local, field, sid in infos:
                row = extract_row(tucker, ri, exp)
                video_recon, U_ic, Vh_ic = recon_fn(row)
                field_raw = raw[local][field].numpy().astype(np.float64)
                # Tucker trajectory: compare against frames 1..200
                traj_raw = field_raw[1:]
                e_traj = rel_err(video_recon, traj_raw)
                # IC SVD: compare against frame 0
                ic_recon = U_ic @ Vh_ic
                e_ic = rel_err(ic_recon, field_raw[0])
                traj_errs.append(e_traj); ic_errs.append(e_ic)
                print(f'  [{os.path.basename(tpath)}#{ri:03d}  {source}  '
                      f'sid={sid}  {field}]  '
                      f'traj={e_traj:.4f}  ic={e_ic:.4f}', flush=True)
            del raw

    traj_errs = np.array(traj_errs)
    ic_errs   = np.array(ic_errs)
    print(f'\n=== [{exp}] N={len(traj_errs)} ===')
    print(f'  Tucker trajectory RelErr: mean={traj_errs.mean():.4f}  '
          f'std={traj_errs.std():.4f}  min={traj_errs.min():.4f}  '
          f'max={traj_errs.max():.4f}')
    print(f'  IC SVD         RelErr: mean={ic_errs.mean():.4f}  '
          f'std={ic_errs.std():.4f}  min={ic_errs.min():.4f}  '
          f'max={ic_errs.max():.4f}')
    return traj_errs, ic_errs


def _unfold(T, mode):
    return np.reshape(np.moveaxis(T, mode, 0), (T.shape[mode], -1))


def _trunc_svd_U(M, k):
    # Always full SVD: faster than scipy.sparse.linalg.svds for small matrices.
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, :k]


def _tucker_hooi_fast(T, rank, n_iter_max=50, tol=1e-6):
    """Fast HOOI for a 3D tensor (float32 + scipy svds + einsum optimize)."""
    ndim = T.ndim
    factors = [_trunc_svd_U(_unfold(T, m), rank[m]) for m in range(ndim)]
    prev_norm = None
    for it in range(n_iter_max):
        for mode in range(ndim):
            Y = T
            for m2 in range(ndim - 1, -1, -1):
                if m2 == mode:
                    continue
                Y = np.tensordot(factors[m2].T, Y, axes=([1], [m2]))
                Y = np.moveaxis(Y, 0, m2)
            factors[mode] = _trunc_svd_U(_unfold(Y, mode), rank[mode])
        core = T
        for mode in range(ndim):
            core = np.tensordot(factors[mode].T, core, axes=([1], [mode]))
            core = np.moveaxis(core, 0, mode)
        cur_norm = float(np.linalg.norm(core))
        if prev_norm is not None and abs(cur_norm - prev_norm) < tol * (cur_norm + 1e-15):
            return core, factors, it + 1
        prev_norm = cur_norm
    return core, factors, n_iter_max


def evaluate_mnist(n_samples: int = 100,
                    rank=(15, 64, 20),
                    seed: int = 0,
                    raw_path: str = '/anvil/projects/x-eng260004/factor_diffusion/original_data/moving_mnist/moving_mnist_20k_2slow.pt'):
    """On-the-fly Tucker HOOI on random videos (no pre-saved factors)."""
    print(f'[mnist] loading {raw_path} ...', flush=True)
    raw = torch.load(raw_path, map_location='cpu', weights_only=False)
    if raw.shape[0] < raw.shape[1]:
        raw = raw.permute(1, 0, 2, 3).contiguous()
    N = raw.shape[0]
    print(f'  raw shape after permute: {tuple(raw.shape)}  N={N}', flush=True)
    print(f'  Tucker rank = {list(rank)}', flush=True)

    rng = random.Random(seed)
    idxs = rng.sample(range(N), min(n_samples, N))

    errs = []
    import time as _t
    t0 = _t.time()
    for i, idx in enumerate(idxs):
        video = raw[idx].numpy().astype(np.float32)   # (T, H, W)
        core, factors, n_iters = _tucker_hooi_fast(video, list(rank))
        U_1, U_2, U_3 = factors
        recon = np.einsum('ijk,ti,hj,wk->thw', core, U_1, U_2, U_3, optimize=True)
        err = float(np.linalg.norm(video - recon) /
                    max(np.linalg.norm(video), 1e-12))
        errs.append(err)
        if (i + 1) % 10 == 0 or (i + 1) == len(idxs):
            elapsed = _t.time() - t0
            print(f'  [{i+1:3d}/{len(idxs)}]  idx={idx:5d}  iters={n_iters:3d}  '
                  f'rel_err={err:.4f}  elapsed={elapsed:.1f}s', flush=True)

    errs = np.array(errs)
    print(f'\n=== [mnist rank={list(rank)}] N={len(errs)} ===')
    print(f'  Tucker recon RelErr: mean={errs.mean():.4f}  '
          f'std={errs.std():.4f}  min={errs.min():.4f}  max={errs.max():.4f}')
    return errs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['burgers', 'karman', 'mnist', 'both', 'all'],
                        default='both')
    parser.add_argument('--n_shards',       type=int, default=10)
    parser.add_argument('--rows_per_shard', type=int, default=10)
    parser.add_argument('--n_samples',      type=int, default=100,
                        help='Only for mnist on-the-fly Tucker.')
    parser.add_argument('--mnist_rank', nargs=3, type=int, default=[15, 64, 20],
                        metavar=('r_T', 'r_H', 'r_W'))
    parser.add_argument('--seed',           type=int, default=0)
    args = parser.parse_args()

    if args.exp == 'both':
        exps = ['burgers', 'karman']
    elif args.exp == 'all':
        exps = ['burgers', 'karman', 'mnist']
    else:
        exps = [args.exp]

    for exp in exps:
        if exp == 'mnist':
            evaluate_mnist(args.n_samples, args.mnist_rank, args.seed)
        else:
            evaluate(exp, args.n_shards, args.rows_per_shard, args.seed)
        print()


if __name__ == '__main__':
    main()
