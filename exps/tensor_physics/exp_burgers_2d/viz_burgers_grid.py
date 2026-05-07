"""
viz_burgers_grid.py — Standalone 3D spatiotemporal heatmap grid visualizer.

Loads N raw Burgers simulation samples from a data-generation shard,
renders each sample's ux and uy fields as 3D spatiotemporal cubes via
exponax, and stitches them into one composite image:

    Layout : N rows × 2 cols   (left = ux | right = uy)
    Colour : shared twilight colormap + colorbar

Usage:
    python viz_burgers_grid.py                        # 4 samples, shard_00000.pt
    python viz_burgers_grid.py --n 4 --shard data_generation/data/shard_00000.pt
    python viz_burgers_grid.py --out my_grid.png --dpi 100
"""

import argparse
import io
import os
import sys

import numpy as np
import torch

# ── exponax path ──────────────────────────────────────────────────────────────
_EXPONAX = '${EXPONAX_ROOT}'
if _EXPONAX not in sys.path:
    sys.path.insert(0, _EXPONAX)


# ---------------------------------------------------------------------------
# Core helpers (extracted from train_burgers_2d.py)
# ---------------------------------------------------------------------------

def _render_to_array(video_np, vlim, resolution=384):
    """
    Render one (T, H, W) float32 ndarray to a uint8 RGB numpy array.
    Uses exponax plot_spatio_temporal_2d with the supplied shared vlim.
    Returns ndarray (h, w, 3) or raises on failure.
    """
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    import exponax as ex
    from exponax.viz._volume import zigzag_alpha
    from functools import partial
    from PIL import Image as PILImage

    trj = jnp.array(video_np[::-1, None, :, :])   # (T, 1, H, W), reversed so T=0 faces viewer
    fig = ex.viz.plot_spatio_temporal_2d(
        trj, vlim=vlim, cmap='twilight',
        bg_color='white', resolution=resolution,
        transfer_function=partial(zigzag_alpha, min_alpha=0.05),
        gamma_correction=2.0,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=72, bbox_inches='tight')
    buf.seek(0)
    arr = np.array(PILImage.open(buf).convert('RGB'))
    plt.close(fig)
    return arr


def render_grid(videos_left, videos_right,
                col_left='ux', col_right='uy',
                suptitle='', resolution=384):
    """
    Stitch N rows × 2 cols into one composite matplotlib Figure.

    videos_left, videos_right : list of (T, H, W) float32 ndarrays
    Returns matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    N = len(videos_left)
    rows = []
    for i in range(N):
        print(f'  rendering sample {i+1}/{N} ...', flush=True)
        vlim_l = float(np.percentile(np.abs(videos_left[i]),  98))
        vlim_r = float(np.percentile(np.abs(videos_right[i]), 98))
        print(f'    {col_left} vlim=({-vlim_l:.4f}, {vlim_l:.4f})  '
              f'{col_right} vlim=({-vlim_r:.4f}, {vlim_r:.4f})')
        arr_l = _render_to_array(videos_left[i],  (-vlim_l, vlim_l), resolution)
        arr_r = _render_to_array(videos_right[i], (-vlim_r, vlim_r), resolution)
        rows.append(np.concatenate([arr_l, arr_r], axis=1))   # (h, 2w, 3)

    canvas = np.concatenate(rows, axis=0)   # (N*h, 2w, 3)
    h_tot, w_tot = canvas.shape[:2]

    fig, ax = plt.subplots(figsize=(w_tot / 72, h_tot / 72 + 0.8), dpi=72)
    ax.imshow(canvas)
    ax.set_title(f'{col_left}  |  {col_right}', fontsize=11)
    ax.axis('off')

    # ── T-direction arrow (depth axis projects ≈ upper-right in vape4d default view)
    # T=0 (initial condition) is the near face; T_max is the far face.
    cell_h = h_tot / N
    cell_w = w_tot / 2
    x0 = cell_w * 0.6          # anchor near bottom-left of first cell
    y0 = cell_h * 0.85
    dx = cell_w * 0.20           # points right-and-up (depth direction)
    dy = -cell_h * 0.13
    ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
    ax.text(x0 + dx * 1.15, y0 + dy * 1.15, 't',
            fontsize=11, color='black', fontweight='bold',
            ha='center', va='center')

    # Colorbar shows colormap shape only (each cube is independently scaled)
    sm = ScalarMappable(cmap='twilight', norm=Normalize(-1, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_ticks([])

    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=0.01)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='3D Burgers heatmap grid')
    parser.add_argument('--shard', type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            'data_generation', 'data', 'shard_00000.pt'),
                        help='Path to data-generation shard .pt file')
    parser.add_argument('--n',          type=int, default=4,
                        help='Number of samples to visualize (default 4)')
    parser.add_argument('--resolution', type=int, default=128,
                        help='exponax voxel resolution (default 128)')
    parser.add_argument('--out',        type=str, default='burgers_grid.png',
                        help='Output image path (default burgers_grid.png)')
    parser.add_argument('--dpi',        type=int, default=120,
                        help='Output DPI (default 120)')
    args = parser.parse_args()

    # ── Load shard ────────────────────────────────────────────────────────────
    print(f'Loading shard: {args.shard}')
    shard = torch.load(args.shard, map_location='cpu', weights_only=False)
    if isinstance(shard, list):
        records = shard
    elif isinstance(shard, dict):
        # packed shard: each value is a (B,...) tensor
        B = next(iter(shard.values())).shape[0]
        records = [{k: v[i] for k, v in shard.items()} for i in range(B)]
    else:
        raise ValueError(f'Unrecognised shard format: {type(shard)}')

    n = min(args.n, len(records))
    print(f'Using {n} samples (shard has {len(records)})')

    videos_ux, videos_uy = [], []
    for i, rec in enumerate(records[:n]):
        ux = rec['ux'].numpy().astype(np.float32)   # (T, H, W)
        uy = rec['uy'].numpy().astype(np.float32)
        nu  = rec.get('nu', '?')
        cd  = rec.get('convection_delta', '?')
        print(f'  sample {i}: ux{ux.shape}  nu={nu:.2e}  cd={cd}')
        videos_ux.append(ux)
        videos_uy.append(uy)

    # ── Render grid ───────────────────────────────────────────────────────────
    print('Rendering 3D heatmap grid ...')
    suptitle = f'n={n} samples from {os.path.basename(args.shard)}'
    fig = render_grid(videos_ux, videos_uy,
                      col_left='ux', col_right='uy',
                      suptitle=suptitle,
                      resolution=args.resolution)

    out_path = args.out
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
