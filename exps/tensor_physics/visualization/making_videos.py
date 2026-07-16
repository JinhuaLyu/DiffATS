"""
making_videos.py — side-by-side comparison of ground truth vs. generated
Karman-vortex 2D vorticity for one randomly selected test sample.

  Left  panel : ground truth   (test_shard_*.pt)
  Right panel : our-method generation (epoch00500_seed0_videos.pt)

The first frame is the conditioning frame; it is held still for a short
"CONDITION" pause at the start, then both panels animate frames 0..199.

Two style variants are produced in a single run:

  1. "vor"     - original yellow/green/black vorticity colormap on a dark
                 background, with the top title bar.
  2. "redblue" - matplotlib RdBu_r diverging colormap on a white background,
                 no top title, larger panel titles and bottom info text,
                 typeset in Nimbus Sans (the URW open-source equivalent of
                 Helvetica, metrically compatible).

Outputs (in /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/output/):
    karman_vortex_gt_vs_gen.gif
    karman_vortex_gt_vs_gen.mp4
    karman_vortex_gt_vs_gen_redblue.gif
    karman_vortex_gt_vs_gen_redblue.mp4

Run:
    module load ffmpeg/4.2.2
    /anvil/projects/x-eng260004/factor_diffusion/diffusion_env/bin/python \
        /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/visualization/making_videos.py
"""

import os
import random
import shutil
import subprocess
import sys

import numpy as np
import torch
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEN_PATH = "/anvil/projects/x-eng260004/factor_diffusion/our_method_generation/" \
           "karman_vortex_2d/epoch00500_seed0_videos.pt"
GT_DIR   = "/anvil/projects/x-eng260004/factor_diffusion/original_data/" \
           "karman_vortex_2d/test_data"
OUT_DIR  = "/home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/output"

SEED              = 0
FPS               = 20
CONDITION_SECONDS = 3.0
N_ANIM_FRAMES     = 200
SAMPLES_PER_SHARD = 50

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

_vor_colors = [
    (1.0, 1.0, 0.0),
    (0.953, 0.490, 0.016),
    (0.0, 0.0, 0.0),
    (0.176, 0.976, 0.529),
    (0.0, 1.0, 1.0),
]
VOR_CMAP    = mcolors.LinearSegmentedColormap.from_list("vor_cmap", _vor_colors)
REDBLUE_CMAP = mcm.get_cmap("RdBu_r")

_LUT_N = 256


def _build_lut(cmap) -> np.ndarray:
    return (cmap(np.linspace(0, 1, _LUT_N)) * 255).astype(np.uint8)


def apply_lut(data: np.ndarray, lut: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    data = np.nan_to_num(data, nan=vmin, posinf=vmax, neginf=vmin)
    idx = np.clip(
        (data - vmin) / (vmax - vmin) * (_LUT_N - 1), 0, _LUT_N - 1
    ).astype(np.int32)
    return lut[idx]


def vor_to_image(vor_frame: np.ndarray, lut: np.ndarray,
                 vmin: float, vmax: float) -> Image.Image:
    """Same orientation as exp_karman_vortex/tucker_karman_demo.py."""
    rgba = apply_lut(vor_frame.T[::-1], lut, vmin, vmax)
    return Image.fromarray(rgba, mode="RGBA").transpose(Image.Transpose.ROTATE_270)

# ---------------------------------------------------------------------------
# Sample selection (shared across both styles)
# ---------------------------------------------------------------------------

print(f"Loading generated tensor:\n  {GEN_PATH}")
gen = torch.load(GEN_PATH, map_location="cpu", weights_only=False)
videos = gen["videos"]
N = videos.shape[0]
print(f"  videos shape: {tuple(videos.shape)}")

rng = random.Random(SEED)
idx = rng.randrange(N)
shard_id = idx // SAMPLES_PER_SHARD
in_shard = idx %  SAMPLES_PER_SHARD
print(f"\nPicked sample idx = {idx}  ->  test_shard_{shard_id:03d}.pt[{in_shard}]")

gt_path = os.path.join(GT_DIR, f"test_shard_{shard_id:03d}.pt")
print(f"Loading ground truth:\n  {gt_path}")
gt_shard = torch.load(gt_path, map_location="cpu", weights_only=False)
gt_clip  = gt_shard[in_shard]

for key in ("niu", "cx", "cy", "r"):
    a = float(gen[key][idx].item())
    b = float(gt_clip[key])
    assert abs(a - b) < 1e-3, f"param mismatch on {key}: gen={a} gt={b}"
print("  parameter check OK")

vor_gt  = gt_clip["vor"].numpy()                  # (201, 128, 128)
vor_gen = videos[idx].numpy()                     # (200, 128, 128)
print(f"  GT  vor shape: {vor_gt.shape}")
print(f"  GEN vor shape: {vor_gen.shape}")

meta = dict(
    Re   = float(gt_clip["Re"]),
    niu  = float(gt_clip["niu"]),
    cx   = int(gt_clip["cx"]),
    cy   = int(gt_clip["cy"]),
    r    = int(gt_clip["r"]),
    step_start = int(gt_clip["step_start"]),
)
print(f"  metadata: {meta}")

sigma = float(vor_gt.std())
VMIN  = max(float(vor_gt.min()), -3.0 * sigma)
VMAX  = min(float(vor_gt.max()),  3.0 * sigma)
print(f"\nColor scale (±3σ of GT): vmin={VMIN:.5f}  vmax={VMAX:.5f}")

# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------


def load_font(candidate_paths, size: int) -> ImageFont.ImageFont:
    for path in candidate_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


DEJAVU_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]
HELVETICA_BOLD = [
    "/usr/share/fonts/urw-base35/NimbusSans-Bold.otf",
] + DEJAVU_BOLD
HELVETICA_REGULAR = [
    "/usr/share/fonts/urw-base35/NimbusSans-Regular.otf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# ---------------------------------------------------------------------------
# Style definitions
# ---------------------------------------------------------------------------

PANEL = 320
GAP   = 16
MARGIN_X = 24

STYLES = [
    # ----- ORIGINAL look ---------------------------------------------------
    dict(
        name="vor",
        out_gif=os.path.join(OUT_DIR, "karman_vortex_gt_vs_gen.gif"),
        out_mp4=os.path.join(OUT_DIR, "karman_vortex_gt_vs_gen.mp4"),
        lut=_build_lut(VOR_CMAP),
        bg=(16, 16, 16, 255),
        fg=(240, 240, 240, 255),
        border=(90, 90, 90, 255),
        show_title=True,
        title_text="Karman Vortex 2D  —  Ground Truth vs. Our Method",
        top_title_h=36,
        panel_title_h=28,
        bottom_info_h=32,
        show_sample_idx=True,
        font_title=load_font(DEJAVU_BOLD, 22),
        font_panel=load_font(DEJAVU_BOLD, 18),
        font_info =load_font(DEJAVU_BOLD, 14),
        font_badge=load_font(DEJAVU_BOLD, 28),
        badge_fill=(220, 60, 60, 230),
        badge_outline=(255, 255, 255, 255),
        badge_text_fill=(255, 255, 255, 255),
    ),
    # ----- NEW red-blue look ----------------------------------------------
    dict(
        name="redblue",
        out_gif=os.path.join(OUT_DIR, "karman_vortex_gt_vs_gen_redblue.gif"),
        out_mp4=os.path.join(OUT_DIR, "karman_vortex_gt_vs_gen_redblue.mp4"),
        lut=_build_lut(REDBLUE_CMAP),
        bg=(255, 255, 255, 255),
        fg=(20, 20, 20, 255),
        border=(170, 170, 170, 255),
        show_title=False,
        title_text="",
        top_title_h=0,
        panel_title_h=42,
        bottom_info_h=40,
        show_sample_idx=False,
        font_title=None,
        font_panel=load_font(HELVETICA_REGULAR, 26),
        font_info =load_font(HELVETICA_REGULAR, 18),
        font_badge=load_font(HELVETICA_BOLD,    26),
        badge_fill=(200, 35, 35, 235),
        badge_outline=(255, 255, 255, 255),
        badge_text_fill=(255, 255, 255, 255),
    ),
]

# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple:
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


def _draw_centered(draw, x_left, y_top, width, height, text, font, fill):
    tw, th = _text_size(draw, text, font)
    draw.text((x_left + (width - tw) // 2, y_top + (height - th) // 2),
              text, fill=fill, font=font)


def render_frame(style, gt_frame, gen_frame, frame_label, condition):
    canvas_w = MARGIN_X * 2 + PANEL * 2 + GAP
    canvas_h = style["top_title_h"] + style["panel_title_h"] + PANEL + style["bottom_info_h"]

    panel_y = style["top_title_h"] + style["panel_title_h"]
    gt_x    = MARGIN_X
    gen_x   = MARGIN_X + PANEL + GAP

    canvas = Image.new("RGBA", (canvas_w, canvas_h), style["bg"])
    draw   = ImageDraw.Draw(canvas)

    if style["show_title"]:
        _draw_centered(draw, 0, 0, canvas_w, style["top_title_h"],
                       style["title_text"], style["font_title"], style["fg"])

    _draw_centered(draw, gt_x,  style["top_title_h"], PANEL, style["panel_title_h"],
                   "Ground Truth", style["font_panel"], style["fg"])
    _draw_centered(draw, gen_x, style["top_title_h"], PANEL, style["panel_title_h"],
                   "Generated (Ours)", style["font_panel"], style["fg"])

    gt_img  = vor_to_image(gt_frame,  style["lut"], VMIN, VMAX).resize(
        (PANEL, PANEL), Image.NEAREST)
    gen_img = vor_to_image(gen_frame, style["lut"], VMIN, VMAX).resize(
        (PANEL, PANEL), Image.NEAREST)
    canvas.paste(gt_img,  (gt_x,  panel_y))
    canvas.paste(gen_img, (gen_x, panel_y))

    for x in (gt_x, gen_x):
        draw.rectangle([x - 1, panel_y - 1, x + PANEL, panel_y + PANEL],
                       outline=style["border"], width=1)

    if condition:
        badge_text = "CONDITION  (t = 0)"
        bw, bh = _text_size(draw, badge_text, style["font_badge"])
        pad_x, pad_y = 18, 10
        box_w = bw + pad_x * 2
        box_h = bh + pad_y * 2
        box_x = (canvas_w - box_w) // 2
        box_y = panel_y + PANEL - box_h - 16
        draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h],
                       fill=style["badge_fill"], outline=style["badge_outline"], width=2)
        draw.text((box_x + pad_x, box_y + pad_y), badge_text,
                  fill=style["badge_text_fill"], font=style["font_badge"])

    info = frame_label
    _draw_centered(draw, 0, canvas_h - style["bottom_info_h"], canvas_w,
                   style["bottom_info_h"], info, style["font_info"], style["fg"])

    return canvas.convert("RGB")


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _find_ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    fallback = "/apps/spack/anvil/apps/ffmpeg/4.2.2-gcc-8.4.1-tuidm5x/bin/ffmpeg"
    if os.path.exists(fallback):
        return fallback
    sys.exit("ERROR: ffmpeg not found. Run `module load ffmpeg/4.2.2` and rerun.")


def save_gif(frames, path: str, fps: int):
    palette_ref = frames[0].quantize(colors=256)
    gif_frames  = [palette_ref] + [f.quantize(palette=palette_ref) for f in frames[1:]]
    duration_ms = int(round(1000 / fps))
    gif_frames[0].save(
        path, save_all=True, append_images=gif_frames[1:],
        duration=duration_ms, loop=0, optimize=False,
    )


def save_mp4(frames, path: str, fps: int, ffmpeg_bin: str):
    w, h = frames[0].size
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "-",
        "-c:v", "mpeg4", "-q:v", "1", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.asarray(f, dtype=np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit(f"ffmpeg exited non-zero while writing {path}")


# ---------------------------------------------------------------------------
# Build and save each style
# ---------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
ffmpeg_bin = _find_ffmpeg()
n_pause    = int(round(CONDITION_SECONDS * FPS))
n_total    = n_pause + N_ANIM_FRAMES
print(f"\nFrames per video: {n_pause} pause + {N_ANIM_FRAMES} animated "
      f"= {n_total}  (@{FPS} fps  =>  {n_total / FPS:.1f} s)")

for style in STYLES:
    print(f"\n=== style: {style['name']} ===")

    frames = []

    condition_img = render_frame(
        style, vor_gt[0], vor_gt[0],
        frame_label=f"frame 000 / {N_ANIM_FRAMES - 1:03d}",
        condition=True,
    )
    frames.extend([condition_img] * n_pause)

    for t in range(N_ANIM_FRAMES):
        img = render_frame(
            style, vor_gt[t], vor_gen[t],
            frame_label=f"frame {t:03d} / {N_ANIM_FRAMES - 1:03d}",
            condition=False,
        )
        frames.append(img)
        if (t + 1) % 50 == 0:
            print(f"  rendered {t + 1}/{N_ANIM_FRAMES}")

    print(f"  saving GIF -> {style['out_gif']}")
    save_gif(frames, style["out_gif"], FPS)
    print(f"    {os.path.getsize(style['out_gif']) / 1e6:.2f} MB")

    print(f"  saving MP4 -> {style['out_mp4']}")
    save_mp4(frames, style["out_mp4"], FPS, ffmpeg_bin)
    print(f"    {os.path.getsize(style['out_mp4']) / 1e6:.2f} MB")

print("\nDone.")
print(f"  Outputs in: {OUT_DIR}")
