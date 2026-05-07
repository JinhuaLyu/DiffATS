#!/usr/bin/env python3
"""
Compute the global std of V_hat from procrustes_refimg shards.

Differences from compute_vhat_stats.py:
  - Reads shards under the
    celebahq128_patchsvd_procrustes_refimg_p8_r16 directory
  - V_hat is computed using the SVD result of the reference image as the
    Procrustes alignment anchor

V_hat columns are approximately unit vectors (from SVD), so each element
has magnitude ~1/sqrt(64) ~= 0.125. We compute a single scalar std over
all images x channels x 64 x 16 elements.

Output:
    {"std": scalar tensor, "mean": scalar tensor}   saved to OUT_PATH
"""

import glob
import os
import torch

SHARD_DIR = "${DATA_ROOT}/tucker_factors/celeba/our_method"
OUT_PATH  = "${DATA_ROOT}/tucker_factors/celeba/our_method/vhat_stats_procrustes_refimg_p32_r32.pt"

def main():
    shard_paths = sorted(glob.glob(
        os.path.join(SHARD_DIR, "celebahq1024_patchsvd_procrustes_refimg_*_shard_*.pt")
    ))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {SHARD_DIR}")
    print(f"Found {len(shard_paths)} shards.")

    all_vals = []
    for p in shard_paths:
        shard = torch.load(p, map_location="cpu", weights_only=False)
        V = shard["V_hat"].float()   # (B, 3, 1024, 16)
        all_vals.append(V.reshape(-1))
        print(f"  {os.path.basename(p)}: V_hat {tuple(V.shape)}")

    all_vals  = torch.cat(all_vals)   # (N_total,)
    vhat_std  = all_vals.std()
    vhat_mean = all_vals.mean()

    print(f"\nV_hat  mean = {vhat_mean:.6f}")
    print(f"V_hat  std  = {vhat_std:.6f}")
    print(f"V_hat  min  = {all_vals.min():.6f}")
    print(f"V_hat  max  = {all_vals.max():.6f}")

    torch.save({"std": vhat_std, "mean": vhat_mean}, OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")

if __name__ == "__main__":
    main()
