import numpy as np
import os
import sys
import torch
from collections import defaultdict
sys.path.insert(0, "${APEBENCH_ROOT}")
import apebench

TOTAL_SAMPLES  = 5000
SAMPLES_PER_PT = 100
BATCH_SIZE     = 10
DG_MIN, DG_MAX = 1.5, 10.0
OUTPUT_DIR     = "${DATA_ROOT}/burgers_2d"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build lookup table
valid = np.load("valid_configs.npy", allow_pickle=True)
dg_discrete_sorted = sorted(set(float(c["diffusion_gamma"]) for c in valid))

valid_pairs = defaultdict(list)   # {dg_discrete: [(cd, ic), ...]}
for c in valid:
    valid_pairs[float(c["diffusion_gamma"])].append(
        (float(c["convection_delta"]), str(c["ic_config"]))
    )

print(f"Number of valid configurations: {len(valid)}")
print(f"Discrete dg values: {dg_discrete_sorted}")
print(f"Continuous sampling range: dg in [{DG_MIN}, {DG_MAX}] (log-uniform)\n")

def get_valid_pairs(dg):
    """Given continuous dg, return conservatively valid (cd, ic) list via floor lookup."""
    candidates = [d for d in dg_discrete_sorted if d <= dg]
    if not candidates:
        return []
    return valid_pairs[max(candidates)]

# Main sampling loop
rng           = np.random.default_rng(seed=42)
sample_buffer = []
shard_idx     = 0
sample_idx    = 0
global_seed   = 0

while sample_idx < TOTAL_SAMPLES:
    # Step 1: log-uniform sampling of continuous dg
    dg = float(np.exp(rng.uniform(np.log(DG_MIN), np.log(DG_MAX))))

    # Step 2: look up the conservatively valid (cd, ic) set for this dg
    pairs = get_valid_pairs(dg)
    if not pairs:
        continue   # dg below all discrete values; resample
    nu = dg / (128 ** 2 * 2 * 2)

    # Step 3: uniformly sample (cd, ic) from the valid set
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
            out = f"{OUTPUT_DIR}/shard_{shard_idx:05d}.pt"
            torch.save(sample_buffer, out)
            print(f"  Saved {out}  (total: {sample_idx})", flush=True)
            sample_buffer = []
            shard_idx += 1

# Flush remaining samples (less than SAMPLES_PER_PT)
if sample_buffer:
    out = f"{OUTPUT_DIR}/shard_{shard_idx:05d}.pt"
    torch.save(sample_buffer, out)
    print(f"  Saved {out}  (total: {sample_idx})", flush=True)
    shard_idx += 1

print(f"\nDone. {sample_idx} samples → {shard_idx} shards in {OUTPUT_DIR}/")
