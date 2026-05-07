"""Generate a 1D Burgers dataset and save it as a single PyTorch ``.pt`` file.

Usage examples
--------------
    # Smoke test: 50 samples on whatever device JAX picks (CPU or GPU)
    python generate_dataset.py --n_samples 50 --batch_size 50 \
        --out_path /work/hdd/bgxp/factor_diffusion/original_data/burgers_1d/smoke_test/burgers_1d_smoke.pt

    # Full run: 10000 samples, ν drawn IID uniformly from 9 fixed values.
    python generate_dataset.py --n_samples 10000 --batch_size 200 \
        --out_path /work/hdd/bgxp/factor_diffusion/original_data/burgers_1d/burgers_1d.pt

Output
------
A torch ``.pt`` file containing a dict with keys:
    tensor    : float32  (N, N_T=201, NX=1024)   trajectories u(t, x)
    nu        : float32  (N,)                    viscosity per sample
    x_coord   : float32  (NX,)                   cell-center x grid
    t_coord   : float32  (N_T,)                  saved time stamps
    init_keys : int32    (N,)                    PRNG seed used per sample
    meta      : dict                             provenance info
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402

from solver import (  # noqa: E402
    DT_SAVE,
    DX,
    FIN_TIME,
    INI_TIME,
    N_INNER,
    N_T,
    NX,
    T_C,
    X_C,
    XL,
    XR,
    init_multi,
    solve_batch,
)

# Nine fixed viscosity values, sampled IID with equal probability per sample.
NU_VALUES = np.array(
    [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1], dtype=np.float32
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n_samples", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--out_path", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[env] JAX devices: {jax.devices()}  (local count = {jax.local_device_count()})")
    print(
        f"[cfg] n_samples={args.n_samples}  batch_size={args.batch_size}  "
        f"seed={args.seed}"
    )
    print(
        f"[cfg] grid: NX={NX}, x in [{XL},{XR}],  N_T={N_T}, t in "
        f"[{INI_TIME},{FIN_TIME}], dt_save={DT_SAVE}, n_inner={N_INNER}"
    )
    print(f"[cfg] viscosity values: {NU_VALUES.tolist()}")

    # 1. Sample viscosities IID (independent, equal probability) per sample.
    rng = np.random.default_rng(args.seed)
    nu_idx = rng.integers(low=0, high=len(NU_VALUES), size=args.n_samples)
    nu_arr = NU_VALUES[nu_idx].astype(np.float32)
    counts = np.bincount(nu_idx, minlength=len(NU_VALUES))
    print("[cfg] per-nu counts:")
    for nu, c in zip(NU_VALUES, counts):
        print(f"        nu = {float(nu):>9.0e} : {int(c):>5d}  ({c / args.n_samples * 100:.2f}%)")

    # 2. Generate per-sample initial conditions in one batched call.
    print("[ic ] sampling initial conditions ...", flush=True)
    t0 = time.time()
    u0_all = np.asarray(
        init_multi(X_C, numbers=int(args.n_samples), k_tot=4, init_key=int(args.seed))
    )
    print(f"[ic ] u0 shape = {u0_all.shape}  ({time.time() - t0:.2f}s)")

    # 3. Solve in batches and accumulate.
    out = np.empty((args.n_samples, N_T, NX), dtype=np.float32)
    t_global = time.time()
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        s, e = b * args.batch_size, min((b + 1) * args.batch_size, args.n_samples)
        eps_b = jnp.asarray(nu_arr[s:e])
        u0_b = jnp.asarray(u0_all[s:e])
        t_b = time.time()
        uu = solve_batch(eps_b, u0_b).block_until_ready()
        out[s:e] = np.asarray(uu, dtype=np.float32)
        elapsed = time.time() - t_b
        eta = (n_batches - b - 1) * (time.time() - t_global) / (b + 1)
        print(
            f"[run] batch {b + 1}/{n_batches}  samples [{s:>5d},{e:>5d})  "
            f"{elapsed:.2f}s  (eta {eta / 60:.1f} min)",
            flush=True,
        )

    print(f"[run] total solver time: {(time.time() - t_global) / 60:.2f} min")

    # 4. Save as a single .pt file with rich metadata.
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tensor": torch.from_numpy(out),
        "nu": torch.from_numpy(nu_arr),
        "nu_index": torch.from_numpy(nu_idx.astype(np.int32)),
        "x_coord": torch.from_numpy(np.array(X_C, dtype=np.float32)),
        "t_coord": torch.from_numpy(np.array(T_C, dtype=np.float32)),
        "init_keys": torch.full((args.n_samples,), int(args.seed), dtype=torch.int32),
        "meta": {
            "equation": "u_t + u u_x = (nu/pi) u_xx  (PDEBench convention)",
            "domain_x": [float(XL), float(XR)],
            "nx": int(NX),
            "domain_t": [float(INI_TIME), float(FIN_TIME)],
            "n_t": int(N_T),
            "dt_save": float(DT_SAVE),
            "n_inner": int(N_INNER),
            "nu_values": NU_VALUES.tolist(),
            "ic_kind": "init_multi(k_tot=4, num_choise_k=2, if_norm=False)",
            "boundary": "periodic",
            "scheme": "MUSCL-VL + upwind Rusanov + central diffusion, "
            "predictor-corrector time update",
            "seed": int(args.seed),
            "n_samples": int(args.n_samples),
            "jax_version": jax.__version__,
            "jax_devices": [str(d) for d in jax.devices()],
        },
    }
    torch.save(payload, out_path)
    size_gb = out_path.stat().st_size / 1024 ** 3
    print(f"[save] wrote {out_path}  ({size_gb:.2f} GB)")
    print("[done]")


if __name__ == "__main__":
    main()
