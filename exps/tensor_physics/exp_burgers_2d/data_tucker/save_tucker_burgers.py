"""
save_tucker_burgers.py — Tucker factor extraction for 2D Burgers equation data.

Pipeline:
  1. Load Burgers shard files (list of dicts: ux/uy (201,128,128) + params)
  2. Tucker decompose ux[1:] and uy[1:] (each (T=200, H=128, W=128)) separately
     → same Tucker rank [r_T, r_H, r_W] for both components
  3. Pick one reference sample (seed=42); Tucker decompose its ux[1:] → ref anchor
  4. Procrustes-align both ux and uy factors to THE SAME U_k_ref (from ux_ref Tucker)
  5. IC SVD at t=0 (rank r_ic):
       ux[0] (128,128): U_ic, S_ic, Vh_ic → align U to U_2_ref, Vh.T to U_3_ref
       uy[0] (128,128): same procedure, same alignment targets (U_2_ref, U_3_ref)
  6. Save {U_1, U_2, U_3, C, U_ic, Vh_ic, nu, cd, dg, ic_config} with 2B rows
     (first B rows from ux, next B rows from uy)

Output directory:
    tucker_burgers_rT{r_T}_rH{r_H}_rW{r_W}/

Usage:
    cd ${REPO_ROOT}/tensor_physics/exp_burgers_2d/data_tucker
    conda run -n rpy2-env python3 save_tucker_burgers.py
    conda run -n rpy2-env python3 save_tucker_burgers.py --n_max 5 --n_workers 2
"""

import argparse
import os
import random
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "BLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
import tensorly as tl
from tensorly.decomposition import tucker
from tqdm import tqdm

_DIR         = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR    = "${DATA_ROOT}/burgers_2d"
SHARD_SIZE   = 100    # physical samples per output shard (→ 2*SHARD_SIZE rows)
REF_SEED     = 42


# ---------------------------------------------------------------------------
# Tucker decomposition
# ---------------------------------------------------------------------------

def tucker_decompose(tensor_f64: np.ndarray, rank: list, n_iter_max: int = 100):
    """HOOI Tucker on (T, H, W). Returns (core, [U_1, U_2, U_3])."""
    core, factors = tucker(
        tl.tensor(tensor_f64),
        rank=rank, n_iter_max=n_iter_max, verbose=False,
    )
    return np.array(core), [np.array(f) for f in factors]


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
    """Orthogonal Q* minimising ||X @ Q - Y||_F  via  SVD(X^T @ Y)."""
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

_ref_factors = None   # [U_1_ref, U_2_ref, U_3_ref] from ux reference Tucker
_U_2_ref     = None   # (H=128, r_ic) — IC SVD U alignment target (= U_2_ref[:, :r_ic])
_rank        = None
_r_ic        = None
_n_iter_max  = None


def _worker_init(ref_factors, U_2_ref, rank, r_ic, n_iter_max):
    global _ref_factors, _U_2_ref, _rank, _r_ic, _n_iter_max
    _ref_factors = ref_factors
    _U_2_ref     = U_2_ref
    _rank        = rank
    _r_ic        = r_ic
    _n_iter_max  = n_iter_max


def _process_one(args):
    i, ux_raw, uy_raw = args
    # ux_raw, uy_raw: (201, 128, 128) float32 numpy arrays

    results = []
    for field_raw in (ux_raw, uy_raw):
        field_f64 = field_raw.astype(np.float64)

        # ── Tucker on frames t=1,...,200 ───────────────────────────────────
        traj = field_f64[1:]                            # (200, 128, 128)
        core, factors = tucker_decompose(traj, _rank, _n_iter_max)
        aligned, Qs = align_factors(factors, _ref_factors)
        U_1, U_2, U_3 = aligned
        Q_1, Q_2, Q_3 = Qs
        C_aligned = np.einsum('abc,ai,bj,ck->ijk', core, Q_1, Q_2, Q_3)

        # ── IC SVD on t=0 ─────────────────────────────────────────────────
        ic = field_f64[0]                               # (128, 128)
        U_ic, S_ic, Vh_ic = svd_truncated(ic, _r_ic)
        # Only align U to U_2_ref; absorb Q_U and S_ic into Vh (like Tucker
        # absorbs factor rotations into the core).
        # This ensures U_ic_stored @ Vh_ic_stored == ic exactly.
        Q_U = procrustes_rotation(U_ic, _U_2_ref)      # align U to U_2_ref
        U_ic_stored  = U_ic @ Q_U                      # (128, r_ic), orthonormal cols
        Vh_ic_stored = Q_U.T @ (S_ic[:, None] * Vh_ic) # (r_ic, 128), rows norm = S_ic
        # Verification: U_ic_stored @ Vh_ic_stored
        #   = (U_ic @ Q_U) @ Q_U.T @ diag(S_ic) @ Vh_ic
        #   = U_ic @ diag(S_ic) @ Vh_ic = ic ✓

        results.append((
            U_1.astype(np.float32),
            U_2.astype(np.float32),
            U_3.astype(np.float32),
            C_aligned.astype(np.float32),
            U_ic_stored.astype(np.float32),
            Vh_ic_stored.astype(np.float32),
        ))

    return (i, results[0], results[1])


# ---------------------------------------------------------------------------
# Shard saving
# ---------------------------------------------------------------------------

def save_shard(shard_idx: int, buf: dict, out_dir: str, rank: list, r_ic: int) -> str:
    path = os.path.join(out_dir, f'tucker_burgers_shard_{shard_idx:04d}.pt')
    torch.save({
        'U_1':       torch.from_numpy(np.stack(buf['U_1'])),    # (2B, 200, r_T)
        'U_2':       torch.from_numpy(np.stack(buf['U_2'])),    # (2B, 128, r_H)
        'U_3':       torch.from_numpy(np.stack(buf['U_3'])),    # (2B, 128, r_W)
        'C':         torch.from_numpy(np.stack(buf['C'])),      # (2B, r_T, r_H, r_W)
        'U_ic':      torch.from_numpy(np.stack(buf['U_ic'])),   # (2B, 128, r_ic)
        'Vh_ic':     torch.from_numpy(np.stack(buf['Vh_ic'])),  # (2B, r_ic, 128)
        'nu':        buf['nu'],
        'cd':        buf['cd'],
        'dg':        buf['dg'],
        'ic_config': buf['ic_config'],
        'sample_idx': buf['sample_idx'],   # physical sample index (for ux: even, uy: odd)
        'rank':  rank,
        'r_ic':  r_ic,
    }, path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str, default=_DATA_DIR,
                        help='Directory containing Burgers shard .pt files')
    parser.add_argument('--rank_T',     type=int, default=5)
    parser.add_argument('--rank_H',     type=int, default=20)
    parser.add_argument('--rank_W',     type=int, default=20)
    parser.add_argument('--r_ic',       type=int, default=20,
                        help='IC SVD rank (must equal rank_H = rank_W for alignment)')
    parser.add_argument('--n_max',      type=int, default=None,
                        help='Process only first N physical samples (debug)')
    parser.add_argument('--shard_size', type=int, default=SHARD_SIZE,
                        help='Physical samples per output shard')
    parser.add_argument('--seed',       type=int, default=REF_SEED)
    parser.add_argument('--n_workers',  type=int, default=os.cpu_count())
    parser.add_argument('--n_iter_max', type=int, default=100)
    parser.add_argument('--out_dir',    type=str, default=None)
    parser.add_argument('--anchor_path', type=str, default=None,
                        help='Path to existing ref_anchor.pt; if set, skip anchor computation')
    parser.add_argument('--shard_pattern', type=str, default='shard_*.pt',
                        help='Glob pattern for input shard files')
    args = parser.parse_args()

    rank = [args.rank_T, args.rank_H, args.rank_W]
    r_ic = args.r_ic
    out_dir = args.out_dir or os.path.join(
        _DIR, f'tucker_burgers_rT{args.rank_T}_rH{args.rank_H}_rW{args.rank_W}'
    )
    os.makedirs(out_dir, exist_ok=True)

    import glob
    import bisect
    shard_files = sorted(glob.glob(os.path.join(args.data_dir, args.shard_pattern)))
    if not shard_files:
        raise FileNotFoundError(f'No shard files found in {args.data_dir}')

    # ── Pass 1: count totals + locate reference sample ────────────────────
    # Load each shard just to count, then immediately free it.
    print(f'Scanning {len(shard_files)} shard file(s) from {args.data_dir} ...')
    shard_offsets = []   # global sample index at start of each shard
    shard_sizes   = []   # number of physical samples in each shard
    n_total_all   = 0
    first_sample  = None
    for p in shard_files:
        shard = torch.load(p, map_location='cpu', weights_only=False)
        shard_offsets.append(n_total_all)
        shard_sizes.append(len(shard))
        n_total_all += len(shard)
        if first_sample is None:
            first_sample = shard[0]   # keep one sample to read metadata
        del shard

    n_total = n_total_all if args.n_max is None else min(n_total_all, args.n_max)

    T_field  = first_sample['ux'].shape[0]   # 201
    H, W     = first_sample['ux'].shape[1], first_sample['ux'].shape[2]
    T_tucker = T_field - 1                    # 200 (skip t=0)
    del first_sample
    print(f'Total physical samples: {n_total_all}  (processing: {n_total})')
    print(f'T_field={T_field}  H={H}  W={W}')
    print(f'Tucker on t=1,...,{T_field-1} → T_tucker={T_tucker}')
    print(f'Tucker rank: {rank}  IC SVD rank: r_ic={r_ic}')
    print(f'Output dir: {os.path.abspath(out_dir)}\n')

    # ── Reference anchor: either load existing or compute from data ───────
    if args.anchor_path is not None:
        print(f'Loading anchor from {args.anchor_path}')
        ref = torch.load(args.anchor_path, map_location='cpu', weights_only=False)
        U_1_ref = ref['U_1_ref'].numpy().astype(np.float64)
        U_2_ref = ref['U_2_ref'].numpy().astype(np.float64)
        U_3_ref = ref['U_3_ref'].numpy().astype(np.float64)
        U_2_ref_ic = ref['U_2_ref_ic'].numpy().astype(np.float64)
        ref_factors = [U_1_ref, U_2_ref, U_3_ref]
        anchor_rank = list(ref.get('rank', rank))
        anchor_r_ic = int(ref.get('r_ic', r_ic))
        assert anchor_rank == rank, f'anchor rank {anchor_rank} != requested {rank}'
        assert anchor_r_ic == r_ic, f'anchor r_ic {anchor_r_ic} != requested {r_ic}'
        print(f'  U_1_ref={U_1_ref.shape}  U_2_ref={U_2_ref.shape}  U_3_ref={U_3_ref.shape}')
        print(f'  U_2_ref_ic={U_2_ref_ic.shape}\n')
    else:
        rng     = random.Random(args.seed)
        ref_idx = rng.randrange(n_total)   # same value as before (same seed + same n_total)
        ref_shard_pos   = bisect.bisect_right(shard_offsets, ref_idx) - 1
        ref_local_idx   = ref_idx - shard_offsets[ref_shard_pos]
        print(f'Reference sample: global={ref_idx}  shard={ref_shard_pos}  local={ref_local_idx}  (seed={args.seed})')

        ref_shard = torch.load(shard_files[ref_shard_pos], map_location='cpu', weights_only=False)
        ref_ux    = ref_shard[ref_local_idx]['ux'].numpy().astype(np.float64)  # (201, 128, 128)
        del ref_shard

        ref_traj = ref_ux[1:].copy()   # (200, 128, 128)
        ref_core, ref_factors = tucker_decompose(ref_traj, rank, args.n_iter_max)
        U_1_ref, U_2_ref, U_3_ref = ref_factors
        print(f'  U_1_ref={U_1_ref.shape}  U_2_ref={U_2_ref.shape}  U_3_ref={U_3_ref.shape}')
        print(f'  C_ref={ref_core.shape}')

        U_2_ref_ic = U_2_ref[:, :r_ic]   # (H, r_ic)

        anchor_path = os.path.join(out_dir, 'ref_anchor.pt')
        torch.save({
            'U_1_ref':        torch.from_numpy(U_1_ref.astype(np.float32)),
            'U_2_ref':        torch.from_numpy(U_2_ref.astype(np.float32)),
            'U_3_ref':        torch.from_numpy(U_3_ref.astype(np.float32)),
            'C_ref':          torch.from_numpy(ref_core.astype(np.float32)),
            'U_2_ref_ic':     torch.from_numpy(U_2_ref_ic.astype(np.float32)),
            'ref_sample_idx': ref_idx,
            'rank': rank, 'r_ic': r_ic,
        }, anchor_path)
        print(f'  Anchor saved → {anchor_path}\n')

    # ── Pass 2: process one input shard at a time ─────────────────────────
    def _empty_buf():
        return {k: [] for k in ('U_1', 'U_2', 'U_3', 'C', 'U_ic', 'Vh_ic',
                                 'nu', 'cd', 'dg', 'ic_config', 'sample_idx')}

    buf = _empty_buf()
    shard_idx_out = 0
    saved_shards  = []
    global_i      = 0          # running physical sample index across all shards
    t0_total      = time.time()
    first_shard_data = None    # for sanity check: keep first output shard's input data

    pbar_total = tqdm(total=n_total, desc='Tucker+SVD+Align', unit='sample', dynamic_ncols=True)

    for shard_file, shard_offset, shard_sz in zip(shard_files, shard_offsets, shard_sizes):
        if global_i >= n_total:
            break

        shard = torch.load(shard_file, map_location='cpu', weights_only=False)
        n_in_shard = min(len(shard), n_total - global_i)

        tasks = [
            (global_i + j, shard[j]['ux'].numpy(), shard[j]['uy'].numpy())
            for j in range(n_in_shard)
        ]
        chunksize = max(1, min(10, n_in_shard // max(1, args.n_workers * 4)))

        with Pool(processes=args.n_workers,
                  initializer=_worker_init,
                  initargs=(ref_factors, U_2_ref_ic, rank, r_ic, args.n_iter_max)) as pool:

            for result in pool.imap(_process_one, tasks, chunksize=chunksize):
                i, res_ux, res_uy = result
                j_local = i - global_i
                sample  = shard[j_local]
                nu, cd, dg, ic = (sample['nu'], sample['convection_delta'],
                                  sample['diffusion_gamma'], sample['ic_config'])

                # Append ux result
                U_1_x, U_2_x, U_3_x, C_x, U_ic_x, Vh_ic_x = res_ux
                buf['U_1'].append(U_1_x);   buf['U_2'].append(U_2_x)
                buf['U_3'].append(U_3_x);   buf['C'].append(C_x)
                buf['U_ic'].append(U_ic_x); buf['Vh_ic'].append(Vh_ic_x)
                buf['nu'].append(nu);       buf['cd'].append(cd)
                buf['dg'].append(dg);       buf['ic_config'].append(ic)
                buf['sample_idx'].append(2 * i)      # ux: even index

                # Append uy result
                U_1_y, U_2_y, U_3_y, C_y, U_ic_y, Vh_ic_y = res_uy
                buf['U_1'].append(U_1_y);   buf['U_2'].append(U_2_y)
                buf['U_3'].append(U_3_y);   buf['C'].append(C_y)
                buf['U_ic'].append(U_ic_y); buf['Vh_ic'].append(Vh_ic_y)
                buf['nu'].append(nu);       buf['cd'].append(cd)
                buf['dg'].append(dg);       buf['ic_config'].append(ic)
                buf['sample_idx'].append(2 * i + 1)  # uy: odd index

                # Flush output shard when full
                if len(buf['U_1']) >= 2 * args.shard_size:
                    p = save_shard(shard_idx_out, buf, out_dir, rank, r_ic)
                    saved_shards.append(p)
                    pbar_total.set_postfix(out_shard=shard_idx_out)
                    shard_idx_out += 1
                    buf = _empty_buf()

                pbar_total.update(1)

        # Save first shard's raw data for sanity check, then free
        if first_shard_data is None:
            first_shard_data = shard[0]
        del shard
        global_i += n_in_shard

    pbar_total.close()

    if buf['U_1']:
        p = save_shard(shard_idx_out, buf, out_dir, rank, r_ic)
        saved_shards.append(p)

    elapsed = time.time() - t0_total
    print(f'\nTotal: {elapsed:.1f}s  ({elapsed / n_total * 1000:.1f} ms/sample  '
          f'workers={args.n_workers})')
    print(f'{len(saved_shards)} shard(s) saved → {out_dir}')

    # ── Write manifest ─────────────────────────────────────────────────────
    manifest_path = os.path.join(out_dir, 'manifest.txt')
    with open(manifest_path, 'w') as f:
        for p in saved_shards:
            f.write(os.path.basename(p) + '\n')
    print(f'Manifest → {manifest_path}')

    # ── Sanity check ───────────────────────────────────────────────────────
    print('\n─── Sanity check (first shard, first entry) ───')
    s = torch.load(saved_shards[0], map_location='cpu', weights_only=False)
    print(f'  Rows in shard: {s["U_1"].shape[0]}  (should be ~{2*args.shard_size})')
    print(f'  U_1 {tuple(s["U_1"].shape)}  U_2 {tuple(s["U_2"].shape)}  '
          f'U_3 {tuple(s["U_3"].shape)}  C {tuple(s["C"].shape)}')
    print(f'  U_ic {tuple(s["U_ic"].shape)}  Vh_ic {tuple(s["Vh_ic"].shape)}')

    U_1 = s['U_1'][0].numpy().astype(np.float64)
    U_2 = s['U_2'][0].numpy().astype(np.float64)
    U_3 = s['U_3'][0].numpy().astype(np.float64)
    C_s = s['C'][0].numpy().astype(np.float64)
    for name, U in [('U_1', U_1), ('U_2', U_2), ('U_3', U_3)]:
        orth = np.linalg.norm(U.T @ U - np.eye(U.shape[1]))
        print(f'  {name}  ||U^TU-I||={orth:.2e}')

    # Tucker reconstruction vs original (use cached first_shard_data)
    sample_idx_0 = s['sample_idx'][0]
    is_ux = (sample_idx_0 % 2 == 0)
    field_orig = (first_shard_data['ux'] if is_ux else first_shard_data['uy']).numpy().astype(np.float64)
    recon  = np.einsum('ijk,ti,hj,wk->thw', C_s, U_1, U_2, U_3)
    recon  = np.clip(recon, field_orig[1:].min(), field_orig[1:].max())
    mse    = float(np.mean((field_orig[1:] - recon) ** 2))
    rel_err = np.linalg.norm(field_orig[1:] - recon) / max(np.linalg.norm(field_orig[1:]), 1e-12)
    print(f'  Tucker recon MSE={mse:.6f}  RelErr={rel_err:.4f}')

    # IC SVD reconstruction check
    U_ic  = s['U_ic'][0].numpy().astype(np.float64)
    Vh_ic = s['Vh_ic'][0].numpy().astype(np.float64)
    ic_recon = U_ic @ Vh_ic
    ic_orig  = field_orig[0]
    ic_rel   = np.linalg.norm(ic_orig - ic_recon) / max(np.linalg.norm(ic_orig), 1e-12)
    print(f'  IC SVD recon  RelErr={ic_rel:.4f}  (expect <0.1 for rank={r_ic})')


if __name__ == '__main__':
    main()
