#!/usr/bin/env python3
"""
Per-(channel, rank) alpha statistics over no-alignment shards.

Output:
  {OUT_PATH}: {
    "std":  (3, R), "mean": (3, R),
    "shard_dir": str, "C": 3, "R": R, "N": N
  }
"""

import glob
import os
import torch
from tqdm import tqdm

SHARD_DIR = "/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/celeba/no_alignment"
OUT_PATH  = "/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/celeba/no_alignment/alpha_stats_no_alignment_p32_r32.pt"


def main():
    shard_paths = sorted(glob.glob(os.path.join(
        SHARD_DIR, "celebahq1024_patchsvd_no_alignment_*_shard_*.pt"
    )))
    if not shard_paths:
        raise FileNotFoundError(f"No no-alignment shards in {SHARD_DIR}")
    print(f"Found {len(shard_paths)} shards")

    sample = torch.load(shard_paths[0], map_location="cpu", weights_only=False)
    B0, C, N, R = sample["alpha"].shape
    print(f"alpha shape per shard: {tuple(sample['alpha'].shape)} -> C={C}, N={N}, R={R}")

    count  = 0
    sum_x  = torch.zeros(C, R, dtype=torch.float64)
    sum_x2 = torch.zeros(C, R, dtype=torch.float64)

    for path in tqdm(shard_paths, desc="Scanning shards"):
        data  = torch.load(path, map_location="cpu", weights_only=False)
        alpha = data["alpha"].double()
        B     = alpha.shape[0]
        alpha_flat = alpha.permute(1, 3, 0, 2).reshape(C, R, -1)
        count  += B * N
        sum_x  += alpha_flat.sum(dim=-1)
        sum_x2 += (alpha_flat ** 2).sum(dim=-1)

    mean = sum_x / count
    var  = sum_x2 / count - mean ** 2
    std  = var.clamp(min=1e-8).sqrt()

    mean_f = mean.float()
    std_f  = std.float()

    print(f"\nPer-channel per-rank statistics (C={C}, R={R}):")
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


if __name__ == "__main__":
    main()
