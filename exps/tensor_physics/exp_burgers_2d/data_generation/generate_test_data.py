import numpy as np
import os
import sys
import torch
from collections import defaultdict
import apebench

TOTAL_SAMPLES  = 500
SAMPLES_PER_PT = 100
BATCH_SIZE     = 10
DG_MIN, DG_MAX = 1.5, 10.0
OUTPUT_DIR     = "/anvil/projects/x-eng260004/factor_diffusion/original_data/burgers_2d/test_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

valid = np.load("valid_configs.npy", allow_pickle=True)
dg_discrete_sorted = sorted(set(float(c["diffusion_gamma"]) for c in valid))

valid_pairs = defaultdict(list)
for c in valid:
    valid_pairs[float(c["diffusion_gamma"])].append(
        (float(c["convection_delta"]), str(c["ic_config"]))
    )

print(f"Number of valid configurations: {len(valid)}")
print(f"Discrete dg values: {dg_discrete_sorted}")
print(f"Continuous sampling range: dg in [{DG_MIN}, {DG_MAX}] (log-uniform)\n")

def get_valid_pairs(dg):
    candidates = [d for d in dg_discrete_sorted if d <= dg]
    if not candidates:
        return []
    return valid_pairs[max(candidates)]

# Different seed from training (42) to get different parameter sequences
rng           = np.random.default_rng(seed=2026)
sample_buffer = []
shard_idx     = 0
sample_idx    = 0
# Large offset ensures no overlap with training global_seed range (0 ~ ~9990)
global_seed   = 1_000_000

while sample_idx < TOTAL_SAMPLES:
    dg = float(np.exp(rng.uniform(np.log(DG_MIN), np.log(DG_MAX))))

    pairs = get_valid_pairs(dg)
    if not pairs:
        continue
    nu = dg / (128 ** 2 * 2 * 2)

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
    global_seed += BATCH_SIZE

    if not np.isfinite(data).all():
        print(f"  [skip] NaN/Inf detected", flush=True)
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
            out = f"{OUTPUT_DIR}/test_shard_{shard_idx:05d}.pt"
            torch.save(sample_buffer, out)
            print(f"  Saved {out}  (total: {sample_idx})", flush=True)
            sample_buffer = []
            shard_idx += 1

if sample_buffer:
    out = f"{OUTPUT_DIR}/test_shard_{shard_idx:05d}.pt"
    torch.save(sample_buffer, out)
    print(f"  Saved {out}  (total: {sample_idx})", flush=True)
    shard_idx += 1

print(f"\nDone. {sample_idx} samples → {shard_idx} shards in {OUTPUT_DIR}/")
