# 1D Burgers data generation

Self-contained generator (no `pdebench` import). Solves
`u_t + u u_x = (nu/pi) u_xx` on `x in [0, 1]` with periodic BCs, matching
PDEBench BurgersEq numerics: 2nd-order MUSCL-VL slope reconstruction +
upwind Rusanov flux + central diffusion + predictor–corrector time stepping.

Output is a single PyTorch `.pt` file with:
- `tensor`  : float32 `(N, 201, 1024)` — trajectories `u(t, x)`
- `nu`      : float32 `(N,)` — viscosity per sample
- `nu_index`: int32   `(N,)` — index into the 9-value list below
- `x_coord` : float32 `(1024,)`
- `t_coord` : float32 `(201,)`
- `init_keys`: int32  `(N,)`
- `meta`    : provenance dict

Viscosity is sampled IID with equal probability from
`{1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1}`.

## Files

- `solver.py` — vendored numerics + JAX `solve_batch(epsilons, u0s)`.
- `generate_dataset.py` — CLI driver.
- `submit_h200.sh` — SLURM script for the DeltaAI `ghx4` partition (GH200).

## Usage

### Smoke test (CPU, ~1 minute for 4 samples)
```bash
module load python/miniforge3_pytorch/2.11.0
PYTHONPATH=/u/jlyu5/.local/lib/python3.12/site-packages \
  python generate_dataset.py --n_samples 4 --batch_size 4 \
    --out_path /work/hdd/bgxp/factor_diffusion/original_data/burgers_1d/smoke_test/burgers_1d_tiny.pt
```

### Full run (GPU, ~30 min for 10000 samples)
```bash
sbatch submit_h200.sh
```

The SLURM script auto-installs `jax[cuda12]==0.10.0` to user site-packages on
first run. Account `bgxp-dtai-gh`, partition `ghx4`, 1 GH200, 1 hour wall time.

### Customize
Edit `submit_h200.sh` to change `N_SAMPLES`, `BATCH_SIZE`, `SEED`, `OUT_PATH`,
or pass them as `python generate_dataset.py ...` arguments directly when
running interactively.

## Loading the dataset
```python
import torch
d = torch.load("/work/hdd/bgxp/factor_diffusion/original_data/burgers_1d/burgers_1d.pt",
               weights_only=False)
trajectories = d["tensor"]   # (10000, 201, 1024)
nu           = d["nu"]       # (10000,)
```
