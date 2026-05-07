"""
save_tucker_karman.py — Tucker factor extraction for Kármán vortex street data.

Pipeline:
  1. Load Kármán shard files (list of dicts: vor (201,128,128) + params)
  2. Tucker decompose vor[1:] (T=200, X=128, Y=128)
     → rank [r_T, r_X, r_Y]
  3. Pick one reference sample (seed=42); Tucker decompose its vor[1:] → ref anchor
  4. Procrustes-align factors to ref anchor
  5. IC SVD at t=0 (rank r_ic):
       vor[0] (128,128): U_ic, S_ic, Vh_ic → align U to U_X_ref[:, :r_ic]
  6. Save {U_T, U_X, U_Y, C, U_ic, Vh_ic, niu, cx, cy, r, Re,
           param_idx, clip_idx, step_start, sample_idx}

Tucker HOOI reuses optimised implementation from tucker_karman_demo.py
(float32 + ARPACK svds + einsum optimize=True).

Output directory:
    tucker_karman_rT{r_T}_rX{r_X}_rY{r_Y}/

Usage:
    python3 save_tucker_karman.py
    python3 save_tucker_karman.py --n_max 5 --n_workers 2   # quick test
"""

import argparse
import glob
import os
import random
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "BLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
from tqdm import tqdm

_DIR      = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_DIR, 'data_generation', 'data')


# ---------------------------------------------------------------------------
# Tucker HOOI (optimised: float32 + ARPACK svds + einsum optimize=True)
# Copied from tucker_karman_demo.py
# ---------------------------------------------------------------------------

def _unfold(T: np.ndarray, mode: int) -> np.ndarray:
    return np.reshape(np.moveaxis(T, mode, 0), (T.shape[mode], -1))


def _trunc_svd_U(M: np.ndarray, k: int) -> np.ndarray:
    min_dim = min(M.shape)
    if k >= min_dim - 1:
        U, _, _ = np.linalg.svd(M, full_matrices=False)
        return U[:, :k]
    from scipy.sparse.linalg import svds
    U, _, _ = svds(M, k=k)
    return U[:, ::-1]


def tucker_hooi(T: np.ndarray, rank: list, n_iter_max: int = 100,
                tol: float = 1e-8):
    """HOOI Tucker decomposition. Works with float32 input."""
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
        if prev_norm is not None:
            if abs(cur_norm - prev_norm) < tol * (cur_norm + 1e-15):
                return core, factors, it + 1
        prev_norm = cur_norm
    return core, factors, n_iter_max


def reconstruct(core, factors):
    return np.einsum('ijk,ai,bj,ck->abc', core, *factors, optimize=True)


# ---------------------------------------------------------------------------
# SVD utilities
# ---------------------------------------------------------------------------

def svd_truncated(M: np.ndarray, r: int):
    """Truncated SVD of M (m×n). Returns U (m,r), S (r,), Vh (r,n)."""
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    return U[:, :r], S[:r], Vh[:r, :]


# ---------------------------------------------------------------------------
# Procrustes alignment
# ---------------------------------------------------------------------------

def procrustes_rotation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Orthogonal Q* minimising ||X @ Q - Y||_F via SVD(X^T @ Y)."""
    U_p, _, Wh_p = np.linalg.svd(X.T @ Y)
    return U_p @ Wh_p


def align_factors(factors, ref_factors):
    """Procrustes-align each factor to its reference counterpart."""
    aligned, Qs = [], []
    for U, U_ref in zip(factors, ref_factors):
        Q = procrustes_rotation(U, U_ref)
        Qs.append(Q)
        aligned.append(U @ Q)
    return aligned, Qs


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

_ref_factors = None
_U_X_ref_ic  = None
_rank        = None
_r_ic        = None
_n_iter_max  = None


def _worker_init(ref_factors, U_X_ref_ic, rank, r_ic, n_iter_max):
    global _ref_factors, _U_X_ref_ic, _rank, _r_ic, _n_iter_max
    _ref_factors = ref_factors
    _U_X_ref_ic  = U_X_ref_ic
    _rank        = rank
    _r_ic        = r_ic
    _n_iter_max  = n_iter_max


def _process_one(args):
    i, vor_np = args
    # vor_np: float32 (201, 128, 128)

    # ── Tucker on frames t=1,...,200 ────────────────────────────────────────
    traj = vor_np[1:]                              # (200, 128, 128) float32
    core, factors, n_iters = tucker_hooi(traj, _rank, _n_iter_max)
    aligned, Qs = align_factors(factors, _ref_factors)
    U_T, U_X, U_Y = aligned
    Q_T, Q_X, Q_Y = Qs
    C_aligned = np.einsum('abc,ai,bj,ck->ijk', core, Q_T, Q_X, Q_Y,
                          optimize=True)

    # ── IC SVD on t=0 ───────────────────────────────────────────────────────
    ic = vor_np[0].astype(np.float64)              # (128, 128)
    U_ic, S_ic, Vh_ic = svd_truncated(ic, _r_ic)
    Q_U = procrustes_rotation(U_ic, _U_X_ref_ic)
    U_ic_stored  = (U_ic @ Q_U).astype(np.float32)
    Vh_ic_stored = (Q_U.T @ (S_ic[:, None] * Vh_ic)).astype(np.float32)

    return (i, n_iters,
            U_T.astype(np.float32),
            U_X.astype(np.float32),
            U_Y.astype(np.float32),
            C_aligned.astype(np.float32),
            U_ic_stored,
            Vh_ic_stored)


# ---------------------------------------------------------------------------
# Shard saving
# ---------------------------------------------------------------------------

def save_shard(shard_idx: int, buf: dict, out_dir: str,
               rank: list, r_ic: int) -> str:
    path = os.path.join(out_dir, f'tucker_karman_shard_{shard_idx:04d}.pt')
    torch.save({
        'U_T':        torch.from_numpy(np.stack(buf['U_T'])),    # (B,200,r_T)
        'U_X':        torch.from_numpy(np.stack(buf['U_X'])),    # (B,128,r_X)
        'U_Y':        torch.from_numpy(np.stack(buf['U_Y'])),    # (B,128,r_Y)
        'C':          torch.from_numpy(np.stack(buf['C'])),      # (B,r_T,r_X,r_Y)
        'U_ic':       torch.from_numpy(np.stack(buf['U_ic'])),   # (B,128,r_ic)
        'Vh_ic':      torch.from_numpy(np.stack(buf['Vh_ic'])),  # (B,r_ic,128)
        'niu':        buf['niu'],
        'cx':         buf['cx'],
        'cy':         buf['cy'],
        'r':          buf['r'],
        'Re':         buf['Re'],
        'param_idx':  buf['param_idx'],
        'clip_idx':   buf['clip_idx'],
        'step_start': buf['step_start'],
        'sample_idx': buf['sample_idx'],
        'rank':       rank,
        'r_ic':       r_ic,
    }, path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Tucker decomposition for Kármán vortex street data')
    parser.add_argument('--data_dir',   type=str, default=_DATA_DIR)
    parser.add_argument('--rank_T',     type=int, default=10)
    parser.add_argument('--rank_X',     type=int, default=128)
    parser.add_argument('--rank_Y',     type=int, default=30)
    parser.add_argument('--r_ic',       type=int, default=30)
    parser.add_argument('--n_max',      type=int, default=None,
                        help='Process only first N clips (debug)')
    parser.add_argument('--shard_size', type=int, default=100)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--n_workers',  type=int, default=os.cpu_count())
    parser.add_argument('--n_iter_max', type=int, default=100)
    parser.add_argument('--out_dir',    type=str, default=None)
    parser.add_argument('--anchor_path', type=str, default=None,
                        help='Path to existing ref_anchor.pt; if set, skip anchor computation')
    parser.add_argument('--shard_pattern', type=str, default='shard_*.pt',
                        help='Glob pattern for input shard files')
    args = parser.parse_args()

    rank  = [args.rank_T, args.rank_X, args.rank_Y]
    r_ic  = args.r_ic
    out_dir = args.out_dir or os.path.join(
        _DIR, f'tucker_karman_rT{args.rank_T}_rX{args.rank_X}_rY{args.rank_Y}'
    )
    os.makedirs(out_dir, exist_ok=True)

    import bisect
    shard_files = sorted(glob.glob(os.path.join(args.data_dir, args.shard_pattern)))
    if not shard_files:
        raise FileNotFoundError(f'No shard files in {args.data_dir}')

    # ── Pass 1: scan all shards (count + record offsets, free each shard) ─
    print(f'Scanning {len(shard_files)} shard file(s) from {args.data_dir} ...')
    shard_offsets = []
    shard_sizes   = []
    n_total_all   = 0
    first_sample  = None
    for p in shard_files:
        shard = torch.load(p, map_location='cpu', weights_only=False)
        shard_offsets.append(n_total_all)
        shard_sizes.append(len(shard))
        n_total_all += len(shard)
        if first_sample is None:
            first_sample = shard[0]
        del shard

    n_total = n_total_all if args.n_max is None else min(n_total_all, args.n_max)

    vor_shape = tuple(first_sample['vor'].shape)
    del first_sample
    print(f'Total clips: {n_total_all}  (processing: {n_total})')
    print(f'vor shape={vor_shape}  |  Tucker on vor[1:] → {(vor_shape[0]-1,)+vor_shape[1:]}')
    print(f'Tucker rank: {rank}  IC SVD rank: r_ic={r_ic}')
    print(f'Output dir: {os.path.abspath(out_dir)}\n')

    # ── Reference anchor: either load existing or compute from data ───────
    if args.anchor_path is not None:
        print(f'Loading anchor from {args.anchor_path}')
        ref = torch.load(args.anchor_path, map_location='cpu', weights_only=False)
        U_T_ref = ref['U_T_ref'].numpy().astype(np.float32)
        U_X_ref = ref['U_X_ref'].numpy().astype(np.float32)
        U_Y_ref = ref['U_Y_ref'].numpy().astype(np.float32)
        U_X_ref_ic = ref['U_X_ref_ic'].numpy().astype(np.float32)
        ref_factors = [U_T_ref, U_X_ref, U_Y_ref]
        anchor_rank = list(ref.get('rank', rank))
        anchor_r_ic = int(ref.get('r_ic', r_ic))
        assert anchor_rank == rank, f'anchor rank {anchor_rank} != requested {rank}'
        assert anchor_r_ic == r_ic, f'anchor r_ic {anchor_r_ic} != requested {r_ic}'
        print(f'  U_T_ref={U_T_ref.shape}  U_X_ref={U_X_ref.shape}  U_Y_ref={U_Y_ref.shape}')
        print(f'  U_X_ref_ic={U_X_ref_ic.shape}\n')
    else:
        rng     = random.Random(args.seed)
        ref_idx = rng.randrange(n_total)
        ref_shard_pos = bisect.bisect_right(shard_offsets, ref_idx) - 1
        ref_local_idx = ref_idx - shard_offsets[ref_shard_pos]
        print(f'Reference clip: global={ref_idx}  shard={ref_shard_pos}  '
              f'local={ref_local_idx}  (seed={args.seed})')

        ref_shard = torch.load(shard_files[ref_shard_pos], map_location='cpu', weights_only=False)
        vor_ref = ref_shard[ref_local_idx]['vor'].numpy()   # float32 (201,128,128)
        del ref_shard

        t0_ref = time.time()
        ref_core, ref_factors, ref_iters = tucker_hooi(
            vor_ref[1:], rank, args.n_iter_max)
        print(f'  Tucker converged in {ref_iters} iters  ({time.time()-t0_ref:.1f}s)')
        U_T_ref, U_X_ref, U_Y_ref = ref_factors
        print(f'  U_T_ref={U_T_ref.shape}  U_X_ref={U_X_ref.shape}  U_Y_ref={U_Y_ref.shape}')

        U_X_ref_ic = U_X_ref[:, :r_ic]                       # (128, r_ic)

        anchor_path = os.path.join(out_dir, 'ref_anchor.pt')
        torch.save({
            'U_T_ref':      torch.from_numpy(U_T_ref.astype(np.float32)),
            'U_X_ref':      torch.from_numpy(U_X_ref.astype(np.float32)),
            'U_Y_ref':      torch.from_numpy(U_Y_ref.astype(np.float32)),
            'C_ref':        torch.from_numpy(ref_core.astype(np.float32)),
            'U_X_ref_ic':   torch.from_numpy(U_X_ref_ic.astype(np.float32)),
            'ref_clip_idx': ref_idx,
            'rank': rank, 'r_ic': r_ic,
        }, anchor_path)
        print(f'  Anchor saved → {anchor_path}\n')

    # ── Pass 2: stream shard-by-shard, Tucker + align + save ──────────────
    def _empty_buf():
        return {k: [] for k in ('U_T', 'U_X', 'U_Y', 'C', 'U_ic', 'Vh_ic',
                                 'niu', 'cx', 'cy', 'r', 'Re',
                                 'param_idx', 'clip_idx', 'step_start',
                                 'sample_idx')}

    buf = _empty_buf()
    shard_idx_out = 0
    saved_shards  = []
    global_i      = 0
    first_shard_data = None
    t0_total = time.time()

    pbar_total = tqdm(total=n_total, desc='Tucker+SVD+Align', unit='clip', dynamic_ncols=True)

    for shard_file, shard_offset, shard_sz in zip(shard_files, shard_offsets, shard_sizes):
        if global_i >= n_total:
            break

        shard = torch.load(shard_file, map_location='cpu', weights_only=False)
        n_in_shard = min(len(shard), n_total - global_i)

        tasks = [(global_i + j, shard[j]['vor'].numpy()) for j in range(n_in_shard)]
        chunksize = max(1, min(10, n_in_shard // max(1, args.n_workers * 4)))

        with Pool(processes=args.n_workers,
                  initializer=_worker_init,
                  initargs=(ref_factors, U_X_ref_ic, rank, r_ic,
                            args.n_iter_max)) as pool:

            for result in pool.imap(_process_one, tasks, chunksize=chunksize):
                i, n_iters, U_T, U_X, U_Y, C, U_ic, Vh_ic = result
                j_local = i - global_i
                clip = shard[j_local]

                buf['U_T'].append(U_T);   buf['U_X'].append(U_X)
                buf['U_Y'].append(U_Y);   buf['C'].append(C)
                buf['U_ic'].append(U_ic); buf['Vh_ic'].append(Vh_ic)
                buf['niu'].append(float(clip.get('niu', 0)))
                buf['cx'].append(int(clip.get('cx', 0)))
                buf['cy'].append(int(clip.get('cy', 0)))
                buf['r'].append(int(clip.get('r', 0)))
                buf['Re'].append(float(clip.get('Re', 0)))
                buf['param_idx'].append(int(clip.get('param_idx', 0)))
                buf['clip_idx'].append(int(clip.get('clip_idx', 0)))
                buf['step_start'].append(int(clip.get('step_start', 0)))
                buf['sample_idx'].append(i)

                if len(buf['U_T']) >= args.shard_size:
                    p = save_shard(shard_idx_out, buf, out_dir, rank, r_ic)
                    saved_shards.append(p)
                    pbar_total.set_postfix(out_shard=shard_idx_out)
                    shard_idx_out += 1
                    buf = _empty_buf()

                pbar_total.update(1)

        if first_shard_data is None:
            first_shard_data = shard[0]
        del shard
        global_i += n_in_shard

    pbar_total.close()

    if buf['U_T']:
        p = save_shard(shard_idx_out, buf, out_dir, rank, r_ic)
        saved_shards.append(p)

    elapsed = time.time() - t0_total
    print(f'\nTotal: {elapsed:.1f}s  ({elapsed / n_total * 1000:.1f} ms/clip  '
          f'workers={args.n_workers})')
    print(f'{len(saved_shards)} shard(s) saved → {out_dir}')

    # ── Manifest ──────────────────────────────────────────────────────────
    manifest_path = os.path.join(out_dir, 'manifest.txt')
    with open(manifest_path, 'w') as f:
        for p in saved_shards:
            f.write(os.path.basename(p) + '\n')
    print(f'Manifest → {manifest_path}')

    # ── Sanity check ──────────────────────────────────────────────────────
    print('\n─── Sanity check (first shard, first entry) ───')
    s = torch.load(saved_shards[0], map_location='cpu', weights_only=False)
    print(f'  Rows in shard: {s["U_T"].shape[0]}  (should be ~{args.shard_size})')
    print(f'  U_T {tuple(s["U_T"].shape)}  U_X {tuple(s["U_X"].shape)}  '
          f'U_Y {tuple(s["U_Y"].shape)}  C {tuple(s["C"].shape)}')
    print(f'  U_ic {tuple(s["U_ic"].shape)}  Vh_ic {tuple(s["Vh_ic"].shape)}')

    U_T0 = s['U_T'][0].numpy().astype(np.float64)
    U_X0 = s['U_X'][0].numpy().astype(np.float64)
    U_Y0 = s['U_Y'][0].numpy().astype(np.float64)
    C0   = s['C'][0].numpy().astype(np.float64)
    for name, U in [('U_T', U_T0), ('U_X', U_X0), ('U_Y', U_Y0)]:
        orth = np.linalg.norm(U.T @ U - np.eye(U.shape[1]))
        print(f'  {name}  ||U^TU-I||={orth:.2e}')

    vor_orig   = first_shard_data['vor'].numpy().astype(np.float64)
    recon      = np.einsum('ijk,ti,xj,yk->txy', C0, U_T0, U_X0, U_Y0, optimize=True)
    rel_err    = (np.linalg.norm(vor_orig[1:] - recon) /
                  max(np.linalg.norm(vor_orig[1:]), 1e-12))
    print(f'  Tucker recon RelErr={rel_err:.4f}')

    U_ic0  = s['U_ic'][0].numpy().astype(np.float64)
    Vh_ic0 = s['Vh_ic'][0].numpy().astype(np.float64)
    ic_recon = U_ic0 @ Vh_ic0
    ic_orig  = vor_orig[0]
    ic_rel   = (np.linalg.norm(ic_orig - ic_recon) /
                max(np.linalg.norm(ic_orig), 1e-12))
    print(f'  IC SVD recon  RelErr={ic_rel:.4f}  (expect <0.1 for r_ic={r_ic})')


if __name__ == '__main__':
    main()
