"""Generate a 1D Reaction-Diffusion dataset (train or test) and save as a
single PyTorch ``.pt`` file.

Same structure as 1d_burgers/data_generation/generate_dataset.py.

Output keys:
    tensor    : float32  (N, N_T=201, NX=1024)   trajectories u(t, x)
    nu        : float32  (N,)                    diffusion coefficient
    rho       : float32  (N,)                    reaction rate
    nu_index  : int32    (N,)                    index into NU_VALUES
    rho_index : int32    (N,)                    index into RHO_VALUES
    x_coord   : float32  (NX,)
    t_coord   : float32  (N_T,)
    init_keys : int32    (N,)                    PRNG seed used per sample
    meta      : dict
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402

from solver_rd import (  # noqa: E402
    DT_INNER,
    DT_SAVE,
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

# Parameter pools (per the user's request).
NU_VALUES = np.array(
    [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1], dtype=np.float32
)
RHO_VALUES = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_path", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[env] JAX devices: {jax.devices()}  (local count = {jax.local_device_count()})")
    print(
        f"[cfg] n_samples={args.n_samples}  batch_size={args.batch_size}  "
        f"seed={args.seed}  out={args.out_path}"
    )
    print(
        f"[cfg] grid: NX={NX}, x in [{XL},{XR}],  N_T={N_T}, t in "
        f"[{INI_TIME},{FIN_TIME}], dt_save={DT_SAVE}, dt_inner={DT_INNER}, "
        f"n_inner_per_save={N_INNER}"
    )
    print(f"[cfg] nu values  ({len(NU_VALUES)}): {NU_VALUES.tolist()}")
    print(f"[cfg] rho values ({len(RHO_VALUES)}): {RHO_VALUES.tolist()}")
    print(f"[cfg] -> {len(NU_VALUES) * len(RHO_VALUES)} combinations, sampled IID per sample")

    # 1. IID sample (nu_idx, rho_idx) per sample.
    rng = np.random.default_rng(args.seed)
    nu_idx = rng.integers(0, len(NU_VALUES), size=args.n_samples).astype(np.int32)
    rho_idx = rng.integers(0, len(RHO_VALUES), size=args.n_samples).astype(np.int32)
    nu_arr = NU_VALUES[nu_idx]
    rho_arr = RHO_VALUES[rho_idx]

    nu_counts = np.bincount(nu_idx, minlength=len(NU_VALUES))
    rho_counts = np.bincount(rho_idx, minlength=len(RHO_VALUES))
    print("[cfg] per-nu counts:")
    for nu, c in zip(NU_VALUES, nu_counts):
        print(f"        nu = {float(nu):>9.0e} : {int(c):>5d}  ({c / args.n_samples * 100:.2f}%)")
    print("[cfg] per-rho counts:")
    for rho, c in zip(RHO_VALUES, rho_counts):
        print(f"        rho = {float(rho):>5.2f}    : {int(c):>5d}  ({c / args.n_samples * 100:.2f}%)")

    # 2. Initial conditions (one batched call; if_norm=True for RD).
    print("[ic ] sampling initial conditions ...", flush=True)
    t0 = time.time()
    u0_all = np.asarray(
        init_multi(
            X_C,
            numbers=int(args.n_samples),
            k_tot=4,
            init_key=int(args.seed),
            num_choise_k=2,
            if_norm=True,
        )
    )
    print(
        f"[ic ] u0 shape = {u0_all.shape}  range = "
        f"[{u0_all.min():.4f}, {u0_all.max():.4f}]  ({time.time() - t0:.2f}s)"
    )

    # 3. Solve in batches.
    out = np.empty((args.n_samples, N_T, NX), dtype=np.float32)
    t_global = time.time()
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        s, e = b * args.batch_size, min((b + 1) * args.batch_size, args.n_samples)
        u0_b = jnp.asarray(u0_all[s:e])
        nu_b = jnp.asarray(nu_arr[s:e])
        rho_b = jnp.asarray(rho_arr[s:e])
        t_b = time.time()
        uu = solve_batch(u0_b, nu_b, rho_b).block_until_ready()
        out[s:e] = np.asarray(uu, dtype=np.float32)
        elapsed = time.time() - t_b
        eta = (n_batches - b - 1) * (time.time() - t_global) / (b + 1)
        print(
            f"[run] batch {b + 1}/{n_batches}  samples [{s:>5d},{e:>5d})  "
            f"{elapsed:.2f}s  (eta {eta / 60:.1f} min)",
            flush=True,
        )
    print(f"[run] total solver time: {(time.time() - t_global) / 60:.2f} min")

    # 4. Save.
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tensor": torch.from_numpy(out),
        "nu": torch.from_numpy(nu_arr),
        "rho": torch.from_numpy(rho_arr),
        "nu_index": torch.from_numpy(nu_idx),
        "rho_index": torch.from_numpy(rho_idx),
        "x_coord": torch.from_numpy(np.array(X_C, dtype=np.float32)),
        "t_coord": torch.from_numpy(np.array(T_C, dtype=np.float32)),
        "init_keys": torch.full((args.n_samples,), int(args.seed), dtype=torch.int32),
        "meta": {
            "equation": "u_t = nu * u_xx + rho * u (1 - u)  (Fisher-KPP, PDEBench convention)",
            "domain_x": [float(XL), float(XR)],
            "nx": int(NX),
            "domain_t": [float(INI_TIME), float(FIN_TIME)],
            "n_t": int(N_T),
            "dt_save": float(DT_SAVE),
            "dt_inner": float(DT_INNER),
            "n_inner": int(N_INNER),
            "nu_values": NU_VALUES.tolist(),
            "rho_values": RHO_VALUES.tolist(),
            "ic_kind": "init_multi(k_tot=4, num_choise_k=2, if_norm=True)",
            "boundary": "periodic",
            "scheme": "Strang splitting: half react (PDEBench piecewise-exact) + "
            "full diffuse (spectral / FFT, exact) + half react",
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
