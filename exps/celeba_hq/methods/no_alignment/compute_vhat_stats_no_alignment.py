#!/usr/bin/env python3
"""
Single scalar V_hat std/mean over no-alignment shards.

V_hat columns are unit vectors -> each element ~ 1/sqrt(d) ~ 0.0312 for d=1024.
Output: {"std": scalar, "mean": scalar}
"""

import glob
import os
import torch

SHARD_DIR = "${DATA_ROOT}/tucker_factors/celeba/no_alignment"
OUT_PATH  = "${DATA_ROOT}/tucker_factors/celeba/no_alignment/vhat_stats_no_alignment_p32_r32.pt"


def main():
    shard_paths = sorted(glob.glob(os.path.join(
        SHARD_DIR, "celebahq1024_patchsvd_no_alignment_*_shard_*.pt"
    )))
    if not shard_paths:
        raise FileNotFoundError(f"No no-alignment shards in {SHARD_DIR}")
    print(f"Found {len(shard_paths)} shards.")

    all_vals = []
    for p in shard_paths:
        shard = torch.load(p, map_location="cpu", weights_only=False)
        V = shard["V_hat"].float()
        all_vals.append(V.reshape(-1))
        print(f"  {os.path.basename(p)}: V_hat {tuple(V.shape)}")

    all_vals  = torch.cat(all_vals)
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
