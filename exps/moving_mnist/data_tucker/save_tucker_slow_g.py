"""
save_tucker_slow_g.py -- Tucker factor extraction for slow-motion Moving MNIST.
                        Saves G = einsum('ijk,hj->ihk', C, U_2) instead of
                        storing U_2 and C separately.

Pipeline:
  1. Load slow_moving_mnist.pt  ->  (N, 20, 64, 64) float64  (no patchification)
  2. Tucker decompose each video (T=20, H=64, W=64) with rank [r_T, r_H, r_W]
  3. Pick one reference video (seed=42); save its factors as ref_anchor.pt
  4. For every video: Procrustes-align U_1, U_2, U_3 to reference
  5. Absorb rotation matrices into core: C_aligned = einsum(C, Q_1, Q_2, Q_3)
  6. Contract U_2 into C_aligned: G[i,h,k] = sum_j C_aligned[i,j,k] * U_2[h,j]
     -> G shape: (r_T, H, r_W)
  7. Save {U_1, G, U_3} into sharded .pt files (U_2 and C not stored separately)

Reconstruction:
  video ~= einsum('ihk,ti,wk->thw', G, U_1, U_3)

Output directory:
    ./tucker_g_r{r_T}_{r_H}_{r_W}/  (under this script's directory)

Usage:
    cd /home/x-jlyu5/jinhua/DiffATS/exps/moving_mnist/data_tucker

    # rank [15, 64, 20]  ->  G shape (B, 15, 64, 20)
    conda run -n rpy2-env python3 save_tucker_slow_g.py --rank_T 15 --rank_H 64 --rank_W 20

    # rank [20, 64, 20]  ->  G shape (B, 20, 64, 20)
    conda run -n rpy2-env python3 save_tucker_slow_g.py --rank_T 20 --rank_H 64 --rank_W 20

    # debug
    conda run -n rpy2-env python3 save_tucker_slow_g.py --rank_T 15 --rank_H 64 --rank_W 20 --n_max 20
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
DATASET_PATH = "/anvil/projects/x-eng260004/factor_diffusion/original_data/moving_mnist/moving_mnist_20k_5slow.pt"
SHARD_SIZE   = 500
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

_ref_factors = None
_rank        = None
_n_iter_max  = None


def _worker_init(ref_factors, rank, n_iter_max):
    global _ref_factors, _rank, _n_iter_max
    _ref_factors = ref_factors
    _rank        = rank
    _n_iter_max  = n_iter_max


def _process_one(args):
    i, frames_u8 = args
    frames_f64 = frames_u8.astype(np.float64)           # (20, 64, 64)
    core, factors = tucker_decompose(frames_f64, _rank, _n_iter_max)
    aligned, Qs = align_factors(factors, _ref_factors)
    U_1_hat, U_2_hat, U_3_hat = aligned
    Q_1, Q_2, Q_3 = Qs
    # Absorb rotations into core: C_aligned = C x_1 Q_1 x_2 Q_2 x_3 Q_3
    C_aligned = np.einsum("abc,ai,bj,ck->ijk", core, Q_1, Q_2, Q_3)
    # Contract U_2 into C: G[i,h,k] = sum_j C[i,j,k] * U_2[h,j]
    # G shape: (r_T, H, r_W)
    G = np.einsum("ijk,hj->ihk", C_aligned, U_2_hat)
    return (i,
            U_1_hat.astype(np.float32),   # (T,  r_T)
            G.astype(np.float32),          # (r_T, H, r_W)
            U_3_hat.astype(np.float32))    # (W,  r_W)


# ---------------------------------------------------------------------------
# Shard saving
# ---------------------------------------------------------------------------

def save_shard(shard_idx: int, buf: dict, out_dir: str, rank: list) -> str:
    path = os.path.join(out_dir, f"tucker_factors_shard_{shard_idx:04d}.pt")
    torch.save({
        "U_1":       torch.from_numpy(np.stack(buf["U_1"])),   # (B, T,   r_T)
        "G":         torch.from_numpy(np.stack(buf["G"])),     # (B, r_T, H, r_W)
        "U_3":       torch.from_numpy(np.stack(buf["U_3"])),   # (B, W,   r_W)
        "video_idx": buf["video_idx"],
        "rank":      rank,
    }, path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank_T',     type=int, default=15,
                        help='Tucker rank for temporal mode (T=20)')
    parser.add_argument('--rank_H',     type=int, default=64,
                        help='Tucker rank for height mode (H=64)')
    parser.add_argument('--rank_W',     type=int, default=20,
                        help='Tucker rank for width mode (W=64)')
    parser.add_argument('--dataset',    type=str, default=DATASET_PATH)
    parser.add_argument('--n_max',      type=int, default=None,
                        help='Process only first N videos (debug)')
    parser.add_argument('--shard_size', type=int, default=SHARD_SIZE)
    parser.add_argument('--seed',       type=int, default=REF_SEED)
    parser.add_argument('--n_workers',  type=int, default=os.cpu_count())
    parser.add_argument('--n_iter_max', type=int, default=100)
    parser.add_argument('--out_dir',    type=str, default=None)
    args = parser.parse_args()

    rank = [args.rank_T, args.rank_H, args.rank_W]
    out_dir = args.out_dir or os.path.join(
        _DIR, f"tucker_g_r{args.rank_T}_{args.rank_H}_{args.rank_W}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset
    print(f"Loading {args.dataset} ...")
    raw  = torch.load(args.dataset, weights_only=True)   # (20, N, 64, 64) uint8
    data = raw.permute(1, 0, 2, 3).numpy()               # (N, 20, 64, 64) uint8
    n_total = len(data) if args.n_max is None else min(len(data), args.n_max)
    data = data[:n_total]
    T, H, W = data.shape[1], data.shape[2], data.shape[3]
    print(f"Using {n_total} videos  |  T={T}  H={H}  W={W}  (no patchification)")
    print(f"Tucker rank: {rank}")
    print(f"G shape per video: ({rank[0]}, {H}, {rank[2]})")
    print(f"Output: {os.path.abspath(out_dir)}")

    # Reference video
    rng     = random.Random(args.seed)
    ref_idx = rng.randrange(n_total)
    print(f"\nReference video: {ref_idx}  (seed={args.seed})")
    ref_frames = data[ref_idx].astype(np.float64)
    ref_core, ref_factors = tucker_decompose(ref_frames, rank, args.n_iter_max)
    U_1_ref, U_2_ref, U_3_ref = ref_factors
    G_ref = np.einsum("ijk,hj->ihk", ref_core, U_2_ref)

    anchor_path = os.path.join(out_dir, 'ref_anchor.pt')
    torch.save({
        'U_1_ref':       torch.from_numpy(U_1_ref.astype(np.float32)),
        'U_2_ref':       torch.from_numpy(U_2_ref.astype(np.float32)),
        'U_3_ref':       torch.from_numpy(U_3_ref.astype(np.float32)),
        'C_ref':         torch.from_numpy(ref_core.astype(np.float32)),
        'G_ref':         torch.from_numpy(G_ref.astype(np.float32)),
        'ref_video_idx': ref_idx,
        'rank':          rank,
    }, anchor_path)
    print(f"  U_1_ref={U_1_ref.shape}  U_2_ref={U_2_ref.shape}  U_3_ref={U_3_ref.shape}")
    print(f"  C_ref={ref_core.shape}  G_ref={G_ref.shape}")
    print(f"  Anchor saved -> {anchor_path}\n")

    # Process all videos in parallel
    buf = {"U_1": [], "G": [], "U_3": [], "video_idx": []}
    shard_idx, saved_shards = 0, []
    chunksize = max(1, min(50, n_total // max(1, args.n_workers * 4)))

    t0_total = time.time()
    with Pool(processes=args.n_workers,
              initializer=_worker_init,
              initargs=(ref_factors, rank, args.n_iter_max)) as pool:
        task_iter = ((i, data[i]) for i in range(n_total))
        pbar = tqdm(pool.imap(_process_one, task_iter, chunksize=chunksize),
                    total=n_total, desc='Tucker+Procrustes+G', unit='video',
                    dynamic_ncols=True)
        for result in pbar:
            i, U_1_hat, G_hat, U_3_hat = result
            buf["U_1"].append(U_1_hat)
            buf["G"].append(G_hat)
            buf["U_3"].append(U_3_hat)
            buf["video_idx"].append(i)
            if len(buf["U_1"]) >= args.shard_size:
                p = save_shard(shard_idx, buf, out_dir, rank)
                saved_shards.append(p)
                pbar.set_postfix(shard=shard_idx)
                shard_idx += 1
                buf = {"U_1": [], "G": [], "U_3": [], "video_idx": []}

    if buf["U_1"]:
        p = save_shard(shard_idx, buf, out_dir, rank)
        saved_shards.append(p)

    elapsed = time.time() - t0_total
    print(f"\nTotal: {elapsed:.1f}s  ({elapsed / n_total * 1000:.1f} ms/video  "
          f"workers={args.n_workers})")
    print(f"{len(saved_shards)} shard(s) saved -> {out_dir}")

    # Write manifest
    manifest_path = os.path.join(out_dir, 'manifest.txt')
    with open(manifest_path, 'w') as f:
        for p in saved_shards:
            f.write(os.path.basename(p) + '\n')
    print(f"Manifest -> {manifest_path}")

    # Sanity check
    print('\n--- Sanity check (first shard, first video) ---')
    s    = torch.load(saved_shards[0], map_location='cpu', weights_only=False)
    U_1  = s['U_1'][0].numpy().astype(np.float64)
    G_s  = s['G'][0].numpy().astype(np.float64)
    U_3  = s['U_3'][0].numpy().astype(np.float64)
    vidx = s['video_idx'][0]
    print(f"  U_1 {U_1.shape}  G {G_s.shape}  U_3 {U_3.shape}")
    for name, U in [('U_1', U_1), ('U_3', U_3)]:
        orth = np.linalg.norm(U.T @ U - np.eye(U.shape[1]))
        print(f"  {name}  ||U^TU-I||={orth:.2e}")
    # Reconstruct: einsum('ihk,ti,wk->thw', G, U_1, U_3)
    recon   = np.einsum('ihk,ti,wk->thw', G_s, U_1, U_3)
    orig_f  = data[vidx].astype(np.float64)
    recon   = np.clip(recon, 0, 255)
    mse     = float(np.mean((orig_f - recon) ** 2))
    psnr    = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
    rel_err = np.linalg.norm(orig_f - recon) / np.linalg.norm(orig_f)
    print(f"  Recon MSE={mse:.2f}  PSNR={psnr:.1f}dB  RelErr={rel_err:.4f}")


if __name__ == '__main__':
    main()
