import numpy as np
import itertools
import sys
sys.path.insert(0, "/home/fzd2816/apebench")
import apebench

TUCKER_RANK = [5, 20, 20]
N_VALIDATE  = 3   # Number of samples to validate per configuration

def tucker_decompose(tensor, rank):
    import tensorly as tl
    from tensorly.decomposition import tucker
    core, factors = tucker(tl.tensor(tensor), rank=rank, n_iter_max=100, verbose=False)
    return np.array(tl.tucker_to_tensor((core, factors)))

def relative_l2(orig, recon):
    denom = np.linalg.norm(orig)
    return float(np.linalg.norm(orig - recon) / denom) if denom > 0 else 0.0

convection_deltas = [-0.1, -0.3, -0.5, -0.8, -1.0]
diffusion_gammas  = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
ic_configs = [
    "fourier;2;true;true",
    "fourier;5;true;true",
    "fourier;8;true;true",
    "grf;1.0;true;true",
    "grf;3.0;true;true",
    "grf;5.0;true;true",
]

total = len(convection_deltas) * len(diffusion_gammas) * len(ic_configs)
valid_configs = []

RESUME_FROM = 144  # Resume from the 145th configuration (0-indexed)

for i, (cd, dg, ic) in enumerate(itertools.product(convection_deltas, diffusion_gammas, ic_configs)):
    if i < RESUME_FROM:
        continue
    print(f"[{i+1}/{total}] cd={cd} dg={dg} ic={ic}", flush=True)
    scenario = apebench.scenarios.difficulty.Burgers(
        num_spatial_dims=2,
        num_points=128,
        convection_delta=cd,
        diffusion_gamma=dg,
        ic_config=ic,
        num_test_samples=N_VALIDATE,
    )
    data = np.array(scenario.get_test_data())  # (N_VALIDATE, 201, 2, 128, 128)
    # Check for numerical divergence (NaN/Inf)
    if not np.isfinite(data).all():
        print(f"  [skip] Numerical divergence (NaN/Inf)", flush=True)
        continue

    l2s = []
    for s in range(N_VALIDATE):
        sample_ux = data[s, :, 0, :, :].astype(np.float64)  # (201, 128, 128)
        recon = tucker_decompose(sample_ux, TUCKER_RANK)
        l2s.append(relative_l2(sample_ux, recon))
    mean_l2 = float(np.mean(l2s))
    status = "[ok]" if mean_l2 <= 0.01 else "[fail]"
    print(f"  {status} Rel-L2={mean_l2:.4f}", flush=True)
    if mean_l2 <= 0.01:
        valid_configs.append(dict(
            convection_delta=cd,
            diffusion_gamma=dg,
            ic_config=ic,
            mean_l2=mean_l2,
        ))

np.save("valid_configs.npy", valid_configs)
print(f"\n{len(valid_configs)}/{total} configurations passed validation; saved to valid_configs.npy")
