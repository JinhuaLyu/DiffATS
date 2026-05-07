import numpy as np
import torch
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import exponax as ex
import glob
import sys

DATA_DIR = "/projects/p32954/jinhua_output/burgers_2d/tucker_factors/tucker_burgers_rT5_rH20_rW20"
OUT_DIR  = "/home/fzd2816/factor_diffusion/tensor_physics/exp_burgers_2d/data_tucker"

# ── 随机选一个 shard 和样本 ───────────────────────────────────────────────────
shards = sorted(glob.glob(f"{DATA_DIR}/tucker_burgers_shard_*.pt"))
if not shards:
    print("No shards found.")
    sys.exit(1)

rng = np.random.default_rng()
shard_path = rng.choice(shards)
shard = torch.load(shard_path, weights_only=False)

n_rows = shard["U_1"].shape[0]
i = rng.integers(n_rows)

U_1  = shard["U_1"][i].numpy().astype(np.float32)   # (200, r_T)
U_2  = shard["U_2"][i].numpy().astype(np.float32)   # (128, r_H)
U_3  = shard["U_3"][i].numpy().astype(np.float32)   # (128, r_W)
C    = shard["C"][i].numpy().astype(np.float32)      # (r_T, r_H, r_W)
U_ic = shard["U_ic"][i].numpy().astype(np.float32)  # (128, r_ic)
Vh_ic= shard["Vh_ic"][i].numpy().astype(np.float32) # (r_ic, 128)

nu         = float(shard["nu"][i])
cd         = float(shard["cd"][i])
dg         = float(shard["dg"][i])
ic_config  = shard["ic_config"][i]
sample_idx = int(shard["sample_idx"][i])
# even sample_idx → ux, odd → uy
field_name = "u_x" if sample_idx % 2 == 0 else "u_y"

print(f"Shard : {shard_path}")
print(f"Row   : {i}  sample_idx={sample_idx}  field={field_name}")
print(f"nu={nu:.2e}  cd={cd:.3f}  dg={dg:.3f}  ic={ic_config}")

# ── 重建速度场 ────────────────────────────────────────────────────────────────
# t=0: initial condition
frame0 = U_ic @ Vh_ic                                        # (128, 128)
# t=1..200: Tucker reconstruction
traj   = np.einsum('ijk,ti,hj,wk->thw', C, U_1, U_2, U_3)  # (200, 128, 128)
field  = np.concatenate([frame0[None], traj], axis=0)        # (201, 128, 128)

print(f"field shape: {field.shape}  range: [{field.min():.4f}, {field.max():.4f}]")

# ── 3D 时空体积渲染 ──────────────────────────────────────────────────────────
trj = jnp.array(field[:, None, :, :])   # (201, 1, 128, 128)
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
fig.colorbar(sm, ax=fig.axes[0], fraction=0.03, pad=0.02, label=field_name)
fig.axes[0].set_title(
    f"{field_name}  (Tucker recon)  sample_idx={sample_idx}\n"
    f"nu={nu:.2e}  cd={cd:.3f}  dg={dg:.3f}  ic={ic_config}",
    fontsize=9,
)

out = f"{OUT_DIR}/viz_tucker_spacetime.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
