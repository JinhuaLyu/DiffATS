import numpy as np
import torch
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import exponax as ex
import glob
import sys

# Pick a random shard and sample
shards = sorted(glob.glob("${DATA_ROOT}/original_data/burgers_2d/test_data/test_shard_*.pt"))
if not shards:
    print("No shards found in data/")
    sys.exit(1)

rng = np.random.default_rng()
shard_path = rng.choice(shards)
shard = torch.load(shard_path, weights_only=False)
sample = shard[rng.integers(len(shard))]

ux = sample["ux"].numpy()   # (201, 128, 128)
uy = sample["uy"].numpy()
nu = sample["nu"]
cd = sample["convection_delta"]
dg = sample["diffusion_gamma"]
ic = sample["ic_config"]

print(f"Shard: {shard_path}")
print(f"ux shape: {ux.shape}, dtype: {ux.dtype}")
print(f"nu={nu:.2e}  cd={cd:.3f}  dg={dg:.3f}  ic={ic}")
print(f"ux range: [{ux.min():.4f}, {ux.max():.4f}]")
print(f"uy range: [{uy.min():.4f}, {uy.max():.4f}]")

# 3D spatio-temporal volume rendering
for arr, name in [(ux, "u_x"), (uy, "u_y")]:
    trj = jnp.array(arr[:, None, :, :])   # (201, 1, 128, 128)
    vlim_abs = float(np.quantile(np.abs(np.array(trj)), 0.8))
    vlim = (-vlim_abs, vlim_abs)

    fig = ex.viz.plot_spatio_temporal_2d(
        trj,
        vlim=vlim,
        cmap="twilight",
        bg_color="black",
        resolution=384,
        distance_scale=100.0,
    )

    sm = mpl.cm.ScalarMappable(
        cmap="twilight",
        norm=mpl.colors.Normalize(vmin=vlim[0], vmax=vlim[1]),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes[0], fraction=0.03, pad=0.02, label=name)
    fig.axes[0].set_title(
        f"{name}  nu={nu:.2e}  cd={cd:.3f}  dg={dg:.3f}\nic={ic}",
        fontsize=9,
    )

    out = f"${REPO_ROOT}/tensor_physics/exp_burgers_2d/data_generation/viz_{name.replace('_', '')}_spacetime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
