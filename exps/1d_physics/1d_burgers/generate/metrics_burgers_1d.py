"""metrics_burgers_1d.py — Per-sample relative-error metrics of generated 1D Burgers
trajectories against the **original physical GT** (not the rank-32 reconstruction).

For each seed file at ``<output_dir>/epoch{epoch:05d}_seed{S}.pt``:
  - Read generated trajectory (N, 1024, 200).
  - Compare against ``original tensor[idx, 1:, :].T`` from
    ``/work/hdd/.../original_data/burgers_1d/burgers_1d_test.pt``.
  - Per-sample L1 rel err, L2 rel err, RMSE.
  - Aggregate to per-seed mean (and sample-wise std).

Across seeds report mean ± std for each metric.
Writes a JSON to ``<output_dir>/metrics_epoch{epoch:05d}.json``.
"""

import argparse
import json
import os
import time

import numpy as np
import torch


DEFAULT_GEN_DIR  = ("${DATA_ROOT}/our_method_generation/"
                    "burgers_1d")
DEFAULT_ORIG_TEST = ("${DATA_ROOT}/original_data/"
                     "burgers_1d/burgers_1d_test.pt")


def per_sample_metrics(traj_gen: np.ndarray, traj_gt: np.ndarray):
    """Both arrays shape (1024, 200). Returns (l1_rel, l2_rel, rmse)."""
    diff = traj_gen - traj_gt
    abs_gt = np.abs(traj_gt).sum()
    l2_gt  = float(np.linalg.norm(traj_gt))
    l1_rel = float(np.abs(diff).sum() / (abs_gt + 1e-12))
    l2_rel = float(np.linalg.norm(diff) / (l2_gt + 1e-12))
    rmse   = float(np.sqrt(np.mean(diff ** 2)))
    return l1_rel, l2_rel, rmse


def load_orig_test(path):
    print(f"[load] orig test: {path}", flush=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensor = payload["tensor"]              # (N_test, 201, 1024) float32
    nu = payload.get("nu")
    print(f"[load]   tensor.shape={tuple(tensor.shape)}  dtype={tensor.dtype}",
          flush=True)
    return tensor, nu


def process_seed(seed_path, orig_tensor):
    print(f"[seed] processing {seed_path}", flush=True)
    d = torch.load(seed_path, map_location="cpu", weights_only=False)
    seed = int(d.get("seed", -1))
    epoch = int(d.get("epoch", -1))
    step  = int(d.get("step",  -1))
    traj_gen_all = d["trajectory"]            # (N, 1024, 200)
    sample_idx   = d["sample_idx"].numpy().astype(np.int64)
    nu_gen       = d.get("nu")

    N = traj_gen_all.shape[0]
    assert traj_gen_all.shape[1] == 1024 and traj_gen_all.shape[2] == 200, \
        f"unexpected gen traj shape {tuple(traj_gen_all.shape)}"

    l1s = np.empty(N, dtype=np.float64)
    l2s = np.empty(N, dtype=np.float64)
    rmses = np.empty(N, dtype=np.float64)

    t0 = time.time()
    for i in range(N):
        ti  = int(sample_idx[i])
        # Original physical GT: skip t=0, transpose (T=200, X=1024) -> (1024, 200)
        v_gt = orig_tensor[ti, 1:, :].numpy().astype(np.float64).T
        v_gen = traj_gen_all[i].numpy().astype(np.float64)
        l1s[i], l2s[i], rmses[i] = per_sample_metrics(v_gen, v_gt)

        if (i + 1) % 100 == 0 or (i + 1) == N:
            print(f"  [seed {seed}] {i+1}/{N}  elapsed={time.time()-t0:.1f}s",
                  flush=True)

    print(f"  [seed {seed}]  mean_L1={l1s.mean():.4e}  "
          f"mean_L2={l2s.mean():.4e}  mean_RMSE={rmses.mean():.4e}",
          flush=True)

    return {
        "seed":     seed,
        "epoch":    epoch,
        "step":     step,
        "N":        int(N),
        "L1_mean":  float(l1s.mean()),
        "L2_mean":  float(l2s.mean()),
        "RMSE_mean":float(rmses.mean()),
        "L1_std":   float(l1s.std()),
        "L2_std":   float(l2s.std()),
        "RMSE_std": float(rmses.std()),
        "L1_per_sample":  l1s.tolist(),
        "L2_per_sample":  l2s.tolist(),
        "RMSE_per_sample":rmses.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir",   type=str, default=DEFAULT_GEN_DIR)
    parser.add_argument("--orig_test", type=str, default=DEFAULT_ORIG_TEST)
    parser.add_argument("--epoch",     type=int, default=1000)
    parser.add_argument("--seeds",     type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--out_json",  type=str, default=None)
    args = parser.parse_args()

    epoch_tag = f"epoch{args.epoch:05d}"
    if args.out_json is None:
        args.out_json = os.path.join(args.gen_dir, f"metrics_{epoch_tag}.json")

    seed_files = [os.path.join(args.gen_dir, f"{epoch_tag}_seed{s}.pt")
                  for s in args.seeds]
    for f in seed_files:
        assert os.path.exists(f), f"Missing: {f}"

    orig_tensor, _ = load_orig_test(args.orig_test)

    seed_results = []
    for f in seed_files:
        seed_results.append(process_seed(f, orig_tensor))

    # Aggregate across seeds
    L1 = np.array([r["L1_mean"] for r in seed_results])
    L2 = np.array([r["L2_mean"] for r in seed_results])
    RM = np.array([r["RMSE_mean"] for r in seed_results])

    summary = {
        "exp":           "burgers_1d",
        "epoch":         args.epoch,
        "seeds":         args.seeds,
        "N_samples":     int(seed_results[0]["N"]),
        "per_seed": [
            {
                "seed": r["seed"],
                "L1_mean":   r["L1_mean"],
                "L2_mean":   r["L2_mean"],
                "RMSE_mean": r["RMSE_mean"],
                "L1_std":    r["L1_std"],
                "L2_std":    r["L2_std"],
                "RMSE_std":  r["RMSE_std"],
            }
            for r in seed_results
        ],
        "across_seeds": {
            "L1_mean":   float(L1.mean()),
            "L1_std":    float(L1.std(ddof=1)),
            "L2_mean":   float(L2.mean()),
            "L2_std":    float(L2.std(ddof=1)),
            "RMSE_mean": float(RM.mean()),
            "RMSE_std":  float(RM.std(ddof=1)),
        },
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n===== burgers_1d  epoch={args.epoch}  N_samples={summary['N_samples']}  "
          f"N_seeds={len(args.seeds)} =====")
    print(f"  L1   per-seed: {[f'{x:.4f}' for x in L1]}")
    print(f"  L1   mean={L1.mean():.4f}   std={L1.std(ddof=1):.4f}")
    print(f"  L2   per-seed: {[f'{x:.4f}' for x in L2]}")
    print(f"  L2   mean={L2.mean():.4f}   std={L2.std(ddof=1):.4f}")
    print(f"  RMSE per-seed: {[f'{x:.4f}' for x in RM]}")
    print(f"  RMSE mean={RM.mean():.4f}   std={RM.std(ddof=1):.4f}")
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
