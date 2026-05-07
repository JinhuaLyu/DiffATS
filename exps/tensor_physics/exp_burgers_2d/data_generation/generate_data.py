import numpy as np
import os
import sys
import torch
from collections import defaultdict
sys.path.insert(0, "/home/fzd2816/apebench")
import apebench

TOTAL_SAMPLES  = 5000
SAMPLES_PER_PT = 100
BATCH_SIZE     = 10
DG_MIN, DG_MAX = 1.5, 10.0
OUTPUT_DIR     = "/projects/p32954/jinhua_data/burgers_2d"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 构建查找表 ────────────────────────────────────────────────────────────────
valid = np.load("valid_configs.npy", allow_pickle=True)
dg_discrete_sorted = sorted(set(float(c["diffusion_gamma"]) for c in valid))

valid_pairs = defaultdict(list)   # {dg_discrete: [(cd, ic), ...]}
for c in valid:
    valid_pairs[float(c["diffusion_gamma"])].append(
        (float(c["convection_delta"]), str(c["ic_config"]))
    )

print(f"有效配置数: {len(valid)}")
print(f"离散 dg 值: {dg_discrete_sorted}")
print(f"连续采样范围: dg ∈ [{DG_MIN}, {DG_MAX}]（对数均匀）\n")

def get_valid_pairs(dg):
    """给定连续 dg，返回保守有效的 (cd, ic) 列表（floor 查找）"""
    candidates = [d for d in dg_discrete_sorted if d <= dg]
    if not candidates:
        return []
    return valid_pairs[max(candidates)]

# ── 采样主循环 ────────────────────────────────────────────────────────────────
rng           = np.random.default_rng(seed=42)
sample_buffer = []
shard_idx     = 0
sample_idx    = 0
global_seed   = 0

while sample_idx < TOTAL_SAMPLES:
    # Step 1：对数均匀采样连续 dg
    dg = float(np.exp(rng.uniform(np.log(DG_MIN), np.log(DG_MAX))))

    # Step 2：查出该 dg 下的保守有效 (cd, ic) 集合
    pairs = get_valid_pairs(dg)
    if not pairs:
        continue   # dg 低于所有离散值，直接重采
    nu = dg / (128 ** 2 * 2 * 2)

    # Step 3：从有效集合中均匀随机选 (cd, ic)
    idx_p = rng.integers(len(pairs))
    cd, ic = pairs[idx_p]

    batch = min(BATCH_SIZE, TOTAL_SAMPLES - sample_idx)
    print(f"[{sample_idx+1}~{sample_idx+batch}/{TOTAL_SAMPLES}] "
          f"cd={cd:.2f} dg={dg:.3f} nu={nu:.2e} ic={ic}", flush=True)

    scenario = apebench.scenarios.difficulty.Burgers(
        num_spatial_dims=2,
        num_points=128,
        convection_delta=cd,
        diffusion_gamma=dg,
        ic_config=ic,
        num_test_samples=batch,
        test_seed=global_seed,
    )
    data = np.array(scenario.get_test_data()).astype(np.float32)
    # data: (batch, 201, 2, 128, 128)
    global_seed += BATCH_SIZE

    if not np.isfinite(data).all():
        print(f"  ✗ NaN/Inf，跳过", flush=True)
        continue

    for b in range(batch):
        sample_buffer.append({
            "ux":               torch.from_numpy(data[b, :, 0, :, :]),
            "uy":               torch.from_numpy(data[b, :, 1, :, :]),
            "nu":               nu,
            "convection_delta": cd,
            "diffusion_gamma":  dg,
            "ic_config":        ic,
        })
        sample_idx += 1

        if len(sample_buffer) == SAMPLES_PER_PT:
            out = f"{OUTPUT_DIR}/shard_{shard_idx:05d}.pt"
            torch.save(sample_buffer, out)
            print(f"  Saved {out}  (total: {sample_idx})", flush=True)
            sample_buffer = []
            shard_idx += 1

# 写入剩余不足 SAMPLES_PER_PT 的样本
if sample_buffer:
    out = f"{OUTPUT_DIR}/shard_{shard_idx:05d}.pt"
    torch.save(sample_buffer, out)
    print(f"  Saved {out}  (total: {sample_idx})", flush=True)
    shard_idx += 1

print(f"\nDone. {sample_idx} samples → {shard_idx} shards in {OUTPUT_DIR}/")
