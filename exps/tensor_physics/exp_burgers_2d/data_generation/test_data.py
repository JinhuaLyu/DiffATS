import numpy as np
import sys
sys.path.insert(0, "${APEBENCH_ROOT}")
import exponax as ex
import matplotlib.pyplot as plt
import jax.numpy as jnp

# 1. Check tensor shapes

import torch
ux_pt = torch.load("data/ux/sample_00000.pt", weights_only=True)
uy_pt = torch.load("data/uy/sample_00000.pt", weights_only=True)
ux = ux_pt["data"].numpy()
uy = uy_pt["data"].numpy()
print(f"nu: {ux_pt['nu']:.4e}")
print(f"ux shape: {ux.shape}, dtype: {ux.dtype}")
print(f"uy shape: {uy.shape}, dtype: {uy.dtype}")
print(f"ux min: {ux.min():.4f}, max: {ux.max():.4f}")
print(f"uy min: {uy.min():.4f}, max: {uy.max():.4f}")

meta_path = "data/metadata.npy"
if __import__("os").path.exists(meta_path):
    meta = np.load(meta_path, allow_pickle=True)
    print(f"metadata[0]: {meta[0]}")
    meta0 = meta[0]
else:
    print("metadata.npy not found, skipping.")
    meta0 = {"nu": float("nan"), "ic_config": "unknown"}

# 2. 3D spatio-temporal volume rendering

import matplotlib as mpl

for channel, arr, name in [("ux", ux, "u_x"), ("uy", uy, "u_y")]:
    trj = jnp.array(arr[:, None, :, :])  # (201, 1, 128, 128)
    trj_np = np.array(trj)
    vlim_abs = float(np.quantile(np.abs(trj_np), 0.8))
    vlim = (-vlim_abs, vlim_abs)

    fig = ex.viz.plot_spatio_temporal_2d(
        trj,
        vlim=vlim,
        cmap="twilight",
        bg_color="black",
        resolution=384,
        distance_scale=100.0,
    )

    # colorbar
    sm = mpl.cm.ScalarMappable(
        cmap="twilight",
        norm=mpl.colors.Normalize(vmin=vlim[0], vmax=vlim[1]),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes[0], fraction=0.03, pad=0.02, label=name)
    fig.axes[0].set_title(f"{name} — sample_00000\nnu={meta0['nu']:.2e}  ic={meta0['ic_config']}")

    out = f"test_{channel}_spacetime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
