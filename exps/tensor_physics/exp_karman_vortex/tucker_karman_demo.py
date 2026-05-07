"""
tucker_karman_demo.py — Tucker decomposition on one randomly selected clip from shard_000.pt.

Loads a random clip, applies Tucker HOOI (rank [10,20,20]),
reconstructs, and saves a side-by-side comparison GIF.

Output: tucker_demo_original_vs_recon.gif  (same directory as this script)
"""

import os
import random
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Tucker HOOI (from tucker_rank_sweep_karman.py)
# ---------------------------------------------------------------------------

def _unfold(T: np.ndarray, mode: int) -> np.ndarray:
    return np.reshape(np.moveaxis(T, mode, 0), (T.shape[mode], -1))


def _trunc_svd_U(M: np.ndarray, k: int) -> np.ndarray:
    min_dim = min(M.shape)
    if k >= min_dim - 1:
        U, _, _ = np.linalg.svd(M, full_matrices=False)
        return U[:, :k]
    from scipy.sparse.linalg import svds
    U, _, _ = svds(M, k=k)
    return U[:, ::-1]


def tucker_hooi(T: np.ndarray, rank: list, n_iter_max: int = 100, tol: float = 1e-8):
    ndim = T.ndim
    factors = [_trunc_svd_U(_unfold(T, m), rank[m]) for m in range(ndim)]
    prev_norm = None
    for it in range(n_iter_max):
        for mode in range(ndim):
            Y = T
            for m2 in range(ndim - 1, -1, -1):
                if m2 == mode:
                    continue
                Y = np.tensordot(factors[m2].T, Y, axes=([1], [m2]))
                Y = np.moveaxis(Y, 0, m2)
            factors[mode] = _trunc_svd_U(_unfold(Y, mode), rank[mode])
        core = T
        for mode in range(ndim):
            core = np.tensordot(factors[mode].T, core, axes=([1], [mode]))
            core = np.moveaxis(core, 0, mode)
        cur_norm = float(np.linalg.norm(core))
        if prev_norm is not None:
            if abs(cur_norm - prev_norm) < tol * (cur_norm + 1e-15):
                return core, factors, it + 1
        prev_norm = cur_norm
    return core, factors, n_iter_max


def reconstruct(core, factors):
    return np.einsum('ijk,ai,bj,ck->abc', core, *factors, optimize=True)


def reconstruction_errors(T, recon):
    mse     = float(np.mean((T - recon) ** 2))
    rel_err = float(np.linalg.norm(T - recon) / (np.linalg.norm(T) + 1e-15))
    return mse, rel_err


def compression_ratio(T, rank):
    original   = T.size
    compressed = int(np.prod(rank)) + sum(T.shape[m] * rank[m] for m in range(T.ndim))
    return original / compressed

# ---------------------------------------------------------------------------
# Visualization (from generate_karman.py)
# ---------------------------------------------------------------------------

_vor_colors = [(1,1,0),(0.953,0.490,0.016),(0,0,0),(0.176,0.976,0.529),(0,1,1)]
VOR_CMAP = mcolors.LinearSegmentedColormap.from_list("vor_cmap", _vor_colors)

_LUT_N = 256
_VOR_LUT = (VOR_CMAP(np.linspace(0, 1, _LUT_N)) * 255).astype(np.uint8)


def _apply_lut(data: np.ndarray, lut: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    data = np.nan_to_num(data, nan=vmin, posinf=vmax, neginf=vmin)
    idx = np.clip((data - vmin) / (vmax - vmin) * (_LUT_N - 1), 0, _LUT_N - 1).astype(np.int32)
    return lut[idx]


def render_frame_pair(vor_orig: np.ndarray, vor_recon: np.ndarray,
                      vmin: float, vmax: float,
                      rank: list, frame_idx: int) -> Image.Image:
    """
    Render original (top) and reconstructed (bottom) vorticity for one frame.
    vor_orig / vor_recon: shape (X, Y).
    """
    orig_rgba  = _apply_lut(vor_orig.T[::-1],  _VOR_LUT, vmin, vmax)
    recon_rgba = _apply_lut(vor_recon.T[::-1], _VOR_LUT, vmin, vmax)

    # 顺时针旋转90度
    img_orig  = Image.fromarray(orig_rgba,  mode="RGBA").transpose(Image.Transpose.ROTATE_270)
    img_recon = Image.fromarray(recon_rgba, mode="RGBA").transpose(Image.Transpose.ROTATE_270)

    ny, nx = orig_rgba.shape[:2]
    label_h = 18
    total_h = (ny + label_h) * 2
    canvas = Image.new("RGBA", (nx, total_h), (20, 20, 20, 255))

    canvas.paste(img_orig,  (0, label_h))
    canvas.paste(img_recon, (0, ny + label_h * 2))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    draw.rectangle([0, 0, nx, label_h - 1], fill=(20, 20, 20, 255))
    draw.text((4, 2), f"Original  (frame {frame_idx:03d})", fill=(255, 255, 255, 255), font=font)

    draw.rectangle([0, ny + label_h, nx, ny + label_h * 2 - 1], fill=(20, 20, 20, 255))
    draw.text((4, ny + label_h + 2),
              f"Tucker r={rank}  (frame {frame_idx:03d})",
              fill=(200, 220, 255, 255), font=font)

    return canvas.convert("RGB")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "data_generation/data/shard_000.pt")
RANK       = [10, 128, 30]
OUT_GIF    = os.path.join(SCRIPT_DIR, "tucker_demo_original_vs_recon.gif")

print(f"Loading {DATA_PATH} ...")
clips   = torch.load(DATA_PATH, map_location='cpu', weights_only=False)
n_clips = len(clips)
print(f"  {n_clips} clips found")

rng_idx = 0
clip    = clips[rng_idx]
meta    = {k: clip[k] for k in ('niu', 'cx', 'cy', 'r', 'Re', 'param_idx', 'clip_idx')
           if k in clip}
print(f"\nSelected clip index: {rng_idx}")
print(f"  Metadata: {meta}")

vor = clip['vor'].numpy()               # float32, shape (T, X, Y)
T_len, X, Y = vor.shape
print(f"  vor shape: {vor.shape}  dtype: {vor.dtype}")
print(f"  vor range: [{vor.min():.5f}, {vor.max():.5f}]")

print(f"\nRunning Tucker HOOI  rank={RANK} ...")
t0 = time.time()
core, factors, n_iters = tucker_hooi(vor, RANK)
elapsed = time.time() - t0
print(f"  Converged in {n_iters} iterations  ({elapsed:.1f}s)")

vor_recon = reconstruct(core, factors)
mse, rel_err = reconstruction_errors(vor, vor_recon)
ratio = compression_ratio(vor, RANK)
print(f"  MSE={mse:.4e}  rel_err={rel_err:.6f}  compression_ratio={ratio:.1f}x")

sigma = vor.std()
vmin  = max(float(vor.min()), -3 * sigma)
vmax  = min(float(vor.max()),  3 * sigma)
print(f"\nColor scale: vmin={vmin:.5f}  vmax={vmax:.5f}  (±3σ)")

print(f"Rendering {T_len} frames ...")
rgb_frames = []
for t in range(T_len):
    rgb_frames.append(render_frame_pair(vor[t], vor_recon[t], vmin, vmax, RANK, t))
    if (t + 1) % 50 == 0:
        print(f"  {t+1}/{T_len}")

# Quantize once, reuse palette for all frames (avoids per-frame k-means)
print("Quantizing palette ...")
palette_ref = rgb_frames[0].quantize(colors=256)
frames = [palette_ref] + [f.quantize(palette=palette_ref) for f in rgb_frames[1:]]

print(f"\nSaving GIF → {OUT_GIF}")
frames[0].save(
    OUT_GIF, save_all=True,
    append_images=frames[1:],
    duration=50, loop=0
)
size_mb = os.path.getsize(OUT_GIF) / 1e6
print(f"  Done  ({size_mb:.1f} MB)")
print(f"\nSummary:")
print(f"  Clip:              {rng_idx}  {meta}")
print(f"  Tucker rank:       {RANK}")
print(f"  dtype:             float32")
print(f"  rel_err:           {rel_err:.6f}")
print(f"  compression_ratio: {ratio:.1f}x")
print(f"  GIF:               {OUT_GIF}")
