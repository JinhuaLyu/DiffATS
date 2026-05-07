"""eval_recon_relerr.py — average rank-32 trajectory reconstruction RelErr over
a batch of samples.

Usage:
    python eval_recon_relerr.py --split train --n 100
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

PATCH_X = 32
PATCH_T = 20
NX = 1024
T_TRAJ = 200
N_BLOCK_X = NX // PATCH_X
N_BLOCK_T = T_TRAJ // PATCH_T

ORIG_DIR = "${DATA_ROOT}/original_data/burgers_1d"
FACTOR_DIR = "${DATA_ROOT}/tucker_factors/burgers_1d"


def reconstruct_traj_batch(alpha: torch.Tensor, V_hat: torch.Tensor) -> torch.Tensor:
    """alpha (B, 320, 32), V_hat (B, 640, 32) -> (B, 1024, 200)."""
    A = alpha @ V_hat.transpose(-1, -2)                                  # (B, 320, 640)
    A = A.reshape(-1, N_BLOCK_X, N_BLOCK_T, PATCH_X, PATCH_T)            # (B, 32, 10, 32, 20)
    A = A.permute(0, 1, 3, 2, 4).contiguous()                            # (B, 32, 32, 10, 20)
    return A.reshape(-1, N_BLOCK_X * PATCH_X, N_BLOCK_T * PATCH_T)       # (B, 1024, 200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for picking sample indices")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"[env] device = {device}")

    orig_name = "burgers_1d.pt" if args.split == "train" else "burgers_1d_test.pt"
    factor_name = f"burgers_1d_{args.split}.pt"

    print(f"[load] {os.path.join(FACTOR_DIR, factor_name)}")
    fac = torch.load(os.path.join(FACTOR_DIR, factor_name), map_location="cpu", weights_only=False)
    alpha_all = fac["alpha"]                # (N, 320, 32)
    V_hat_all = fac["V_hat"]                # (N, 640, 32)
    N = alpha_all.shape[0]
    n = min(args.n, N)
    print(f"[load] factors N={N}  picking n={n}  seed={args.seed}")

    rng = np.random.default_rng(args.seed)
    idx_list = rng.choice(N, size=n, replace=False)
    idx_list = np.sort(idx_list)
    print(f"[idx ] first 10 = {idx_list[:10].tolist()} ...")

    print(f"[load] {os.path.join(ORIG_DIR, orig_name)}  (large; this may take a moment)")
    t0 = time.time()
    orig = torch.load(os.path.join(ORIG_DIR, orig_name), map_location="cpu", weights_only=False)
    print(f"[load] original tensor.shape={tuple(orig['tensor'].shape)}  ({time.time()-t0:.1f}s)")

    # Gather selected samples
    traj_orig = orig["tensor"][torch.from_numpy(idx_list).long()][:, 1:, :]  # (n, 200, 1024)
    traj_orig = traj_orig.transpose(-1, -2).contiguous().to(device, dtype=torch.float64)  # (n, 1024, 200)

    alpha = alpha_all[torch.from_numpy(idx_list).long()].to(device, dtype=torch.float64)
    V_hat = V_hat_all[torch.from_numpy(idx_list).long()].to(device, dtype=torch.float64)
    traj_recon = reconstruct_traj_batch(alpha, V_hat)                          # (n, 1024, 200)

    diff = (traj_orig - traj_recon).reshape(n, -1)
    base = traj_orig.reshape(n, -1)
    per_sample_rel = (diff.norm(dim=1) / base.norm(dim=1).clamp(min=1e-12)).cpu().numpy()

    print(f"\n[eval] split={args.split}  n={n}")
    print(f"[eval] per-sample RelErr  mean = {per_sample_rel.mean():.4e}")
    print(f"[eval]                    std  = {per_sample_rel.std():.4e}")
    print(f"[eval]                    min  = {per_sample_rel.min():.4e}")
    print(f"[eval]                    max  = {per_sample_rel.max():.4e}")
    print(f"[eval]                    median = {np.median(per_sample_rel):.4e}")


if __name__ == "__main__":
    main()
