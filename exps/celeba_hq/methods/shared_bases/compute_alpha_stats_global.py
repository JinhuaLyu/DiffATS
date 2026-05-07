#!/usr/bin/env python3
"""
Per-(channel, rank) statistics over alpha shards from the GLOBAL-V (shared
basis) preprocessing pipeline.

Input shards layout (produced by all_save_global_pca.py):
  {SHARD_DIR}/celebahq1024_global_pca_p32_r32_shard_XXXX.pt:
    {
      "alpha":     (B, 3, 1024, 32),
      "filenames": [...],
      "patch":     32,
      "rank":      32,
    }

Output:
  {OUT_PATH}: {
    "std":       (3, R)   per-channel per-rank std  (use for normalization)
    "mean":      (3, R)   per-channel per-rank mean
    "shard_dir": str
    "C": 3, "R": R, "N": N
  }

Normalization at training:
    data    = alpha   / std[None, :, None, :]   # (B,3,N,R) / (1,3,1,R)
    alpha   = samples * std[None, :, None, :]   # after sampling
"""

import glob
import os
import torch
from tqdm import tqdm

# ---------------------------------------------
# Config
# ---------------------------------------------
SHARD_DIR = "${DATA_ROOT}/tucker_factors/celeba/shared_bases"
OUT_PATH  = "${DATA_ROOT}/tucker_factors/celeba/shared_bases/alpha_stats_global_pca_p32_r32.pt"


def main():
    shard_paths = sorted(glob.glob(os.path.join(
        SHARD_DIR, "celebahq1024_global_pca_*_shard_*.pt"
    )))
    if not shard_paths:
        raise FileNotFoundError(f"No global-PCA shards found in {SHARD_DIR}")
    print(f"Found {len(shard_paths)} shards")

    sample_shard = torch.load(shard_paths[0], map_location="cpu", weights_only=False)
    B0, C, N, R = sample_shard["alpha"].shape
    print(f"alpha shape per shard: {tuple(sample_shard['alpha'].shape)}  ->  C={C}, N={N}, R={R}")

    # Welford-style two-pass accumulation: count, sum, sum-of-squares per (C, R).
    # Treat (B, N) as i.i.d. samples for each (channel, rank) scalar.
    count  = 0
    sum_x  = torch.zeros(C, R, dtype=torch.float64)
    sum_x2 = torch.zeros(C, R, dtype=torch.float64)

    for path in tqdm(shard_paths, desc="Scanning shards"):
        data  = torch.load(path, map_location="cpu", weights_only=False)
        alpha = data["alpha"].double()                         # (B, 3, N, R)
        B     = alpha.shape[0]
        # (B, C, N, R) -> (C, R, B*N)
        alpha_flat = alpha.permute(1, 3, 0, 2).reshape(C, R, -1)

        count  += B * N
        sum_x  += alpha_flat.sum(dim=-1)
        sum_x2 += (alpha_flat ** 2).sum(dim=-1)

    mean = sum_x  / count
    var  = sum_x2 / count - mean ** 2
    std  = var.clamp(min=1e-8).sqrt()

    mean_f = mean.float()
    std_f  = std.float()

    print(f"\nPer-channel per-rank statistics  (shape: C={C}, R={R}):")
    for c, ch in enumerate("RGB"):
        print(f"  Channel {ch}: mean range [{mean_f[c].min():.4f}, {mean_f[c].max():.4f}]"
              f"   std range [{std_f[c].min():.4f}, {std_f[c].max():.4f}]")
    print(f"\n  Overall std: min={std_f.min():.4f}  max={std_f.max():.4f}  mean={std_f.mean():.4f}")

    torch.save({
        "std":       std_f,
        "mean":      mean_f,
        "shard_dir": SHARD_DIR,
        "C": C, "R": R, "N": N,
    }, OUT_PATH)
    print(f"\nSaved -> {OUT_PATH}")
    print(f"  std  shape: {tuple(std_f.shape)}")
    print(f"  mean shape: {tuple(mean_f.shape)}")
    print(f"\nUsage in training:")
    print(f"  data    = alpha   / std[None, :, None, :]   # (B,3,{N},{R}) / (1,3,1,{R})")
    print(f"  samples = samples * std[None, :, None, :]   # after sampling")


if __name__ == "__main__":
    main()
