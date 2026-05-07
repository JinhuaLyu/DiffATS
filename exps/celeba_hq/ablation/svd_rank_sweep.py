"""
CelebA 1024x1024 Channel-wise SVD Rank Sweep
Two variants compared side-by-side:
  (A) With patchification:  each channel -> (n_patches=1024, patch_pixels=1024) matrix
  (B) Without patchification: each channel -> raw (H=1024, W=1024) matrix
Both use SVD rank-r approximation for r in [16, 32, 64, 128].
Metrics are averaged over all images in DATA_DIR.
"""

import os
import glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------
# Paths
# ----------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "celeba_1024")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "svd_results")
os.makedirs(OUT_DIR, exist_ok=True)

PATCH_SIZE = 32
RANKS      = [16, 32, 64, 128]
MAX_PIXEL  = 255.0


# ----------------------------------------------
# Patchify / Unpatchify
# ----------------------------------------------
def patchify(img_c: np.ndarray, p: int) -> np.ndarray:
    """img_c: (H, W) -> (n_patches, p*p)"""
    H, W = img_c.shape
    nH, nW = H // p, W // p
    return (img_c
            .reshape(nH, p, nW, p)
            .transpose(0, 2, 1, 3)
            .reshape(nH * nW, p * p))


def unpatchify(M: np.ndarray, p: int, H: int, W: int) -> np.ndarray:
    """M: (n_patches, p*p) -> (H, W)"""
    nH, nW = H // p, W // p
    return (M
            .reshape(nH, nW, p, p)
            .transpose(0, 2, 1, 3)
            .reshape(H, W))


# ----------------------------------------------
# SVD low-rank approximation
# ----------------------------------------------
def svd_approx(M: np.ndarray, rank: int) -> np.ndarray:
    """M: (m, n) -> rank-r approximation (m, n)"""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U[:, :rank] @ (s[:rank, None] * Vt[:rank, :])


def psnr(rmse: float, max_val: float = MAX_PIXEL) -> float:
    if rmse == 0:
        return float("inf")
    return 20.0 * np.log10(max_val / rmse)


# ----------------------------------------------
# Rank sweep for one variant
# ----------------------------------------------
def run_sweep(matrices: np.ndarray, img_shape, use_patch: bool):
    """
    matrices : (C, m, n)  -- per-channel matrices already in desired form
    img_shape: (H, W, C)
    use_patch: whether matrices came from patchification (needed for unpatchify)
    Returns:
      results        : dict rank -> {"rmse": float, "rel_err": float, "psnr": float}
      reconstructions: dict rank -> uint8 image (H, W, C)
    """
    H, W, C = img_shape
    results, reconstructions = {}, {}

    for rank in RANKS:
        rmse_list, rel_err_list, rec_channels = [], [], []

        for c in range(C):
            M_c = matrices[c]
            M_r = svd_approx(M_c, rank)

            diff = M_c - M_r
            rmse_list.append(np.sqrt(np.mean(diff ** 2)))
            rel_err_list.append(np.linalg.norm(diff, "fro") / np.linalg.norm(M_c, "fro"))

            if use_patch:
                rec_c = unpatchify(M_r, PATCH_SIZE, H, W)
            else:
                rec_c = M_r  # already (H, W)
            rec_channels.append(rec_c)

        rec_img = np.clip(np.stack(rec_channels, axis=-1), 0, 255).astype(np.uint8)
        reconstructions[rank] = rec_img
        mean_rmse = float(np.mean(rmse_list))
        results[rank] = {
            "rmse":    mean_rmse,
            "rel_err": float(np.mean(rel_err_list)),
            "psnr":    psnr(mean_rmse),
        }

    return results, reconstructions


# ----------------------------------------------
# Main
# ----------------------------------------------
def main():
    img_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")) +
                       glob.glob(os.path.join(DATA_DIR, "*.png")))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {DATA_DIR}")
    print(f"Found {len(img_paths)} images in {DATA_DIR}\n")

    # Accumulators: rank -> {"rmse": [], "rel_err": [], "psnr": []} for each variant
    accum_patch = {r: {"rmse": [], "rel_err": [], "psnr": []} for r in RANKS}
    accum_raw   = {r: {"rmse": [], "rel_err": [], "psnr": []} for r in RANKS}

    # Keep reconstructions from the first image for the comparison figure
    first_rec_patch = first_rec_raw = first_img = None

    for idx, img_path in enumerate(img_paths):
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        H, W, C = img.shape

        patch_matrices = np.stack([patchify(img[:, :, c], PATCH_SIZE) for c in range(C)], axis=0)
        raw_matrices   = np.stack([img[:, :, c] for c in range(C)], axis=0)

        res_patch, rec_patch = run_sweep(patch_matrices, (H, W, C), use_patch=True)
        res_raw,   rec_raw   = run_sweep(raw_matrices,   (H, W, C), use_patch=False)

        for rank in RANKS:
            for key in ("rmse", "rel_err", "psnr"):
                accum_patch[rank][key].append(res_patch[rank][key])
                accum_raw[rank][key].append(res_raw[rank][key])

        if idx == 0:
            first_rec_patch = rec_patch
            first_rec_raw   = rec_raw
            first_img       = img

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Processed {idx + 1}/{len(img_paths)}: {os.path.basename(img_path)}")

    # Average metrics
    avg_patch = {r: {k: float(np.mean(v)) for k, v in accum_patch[r].items()} for r in RANKS}
    avg_raw   = {r: {k: float(np.mean(v)) for k, v in accum_raw[r].items()}   for r in RANKS}

    # Save per-rank reconstruction PNGs (from first image)
    for rank in RANKS:
        Image.fromarray(first_rec_patch[rank]).save(
            os.path.join(OUT_DIR, f"patch_rank_{rank:03d}.png"))
        Image.fromarray(first_rec_raw[rank]).save(
            os.path.join(OUT_DIR, f"nopatch_rank_{rank:03d}.png"))

    # Print & save metrics
    hdr = f"{'Rank':>6}  {'RMSE':>8}  {'PSNR(dB)':>9}  {'RelErr':>10}"
    sep = "-" * 40

    print(f"\n-- With patchification (avg over {len(img_paths)} images) --")
    print(hdr); print(sep)
    for rank in RANKS:
        r = avg_patch[rank]
        print(f"{rank:>6}  {r['rmse']:>8.4f}  {r['psnr']:>9.4f}  {r['rel_err']:>10.6f}")

    print(f"\n-- Without patchification (avg over {len(img_paths)} images) --")
    print(hdr); print(sep)
    for rank in RANKS:
        r = avg_raw[rank]
        print(f"{rank:>6}  {r['rmse']:>8.4f}  {r['psnr']:>9.4f}  {r['rel_err']:>10.6f}")

    metrics_path = os.path.join(OUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Averaged over {len(img_paths)} images from {DATA_DIR}\n\n")
        col_w = "  {'---With patch---':>32}  {'---No patch---':>32}"
        f.write(f"{'Rank':>6}  {'RMSE':>8}  {'PSNR(dB)':>9}  {'RelErr':>10}"
                f"  {'RMSE':>8}  {'PSNR(dB)':>9}  {'RelErr':>10}\n")
        f.write("-" * 76 + "\n")
        for rank in RANKS:
            rp = avg_patch[rank]
            rn = avg_raw[rank]
            f.write(
                f"{rank:>6}  {rp['rmse']:>8.4f}  {rp['psnr']:>9.4f}  {rp['rel_err']:>10.6f}"
                f"  {rn['rmse']:>8.4f}  {rn['psnr']:>9.4f}  {rn['rel_err']:>10.6f}\n"
            )
    print(f"\nMetrics -> {metrics_path}")

    # Comparison figure (first image)
    orig_uint8 = first_img.astype(np.uint8)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(
        f"CelebA SVD Rank Sweep -- With vs Without Patchification (32x32)\n"
        f"Metrics averaged over {len(img_paths)} images",
        fontsize=12
    )
    row_labels = ["With patch\n(1024x1024 matrix)", "No patch\n(1024x1024 matrix)"]
    row_data   = [(avg_patch, first_rec_patch), (avg_raw, first_rec_raw)]

    for row, (label, (avg, rec)) in enumerate(zip(row_labels, row_data)):
        axes[row, 0].imshow(orig_uint8)
        axes[row, 0].set_title(f"Original\n({label})", fontsize=9)
        axes[row, 0].axis("off")
        for col, rank in enumerate(RANKS, start=1):
            r = avg[rank]
            axes[row, col].imshow(rec[rank])
            axes[row, col].set_title(
                f"Rank {rank}\nRMSE={r['rmse']:.2f}  PSNR={r['psnr']:.2f}dB\nRelErr={r['rel_err']:.4f}",
                fontsize=8
            )
            axes[row, col].axis("off")

    plt.tight_layout()
    comp_path = os.path.join(OUT_DIR, "comparison.png")
    plt.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison figure -> {comp_path}")


if __name__ == "__main__":
    main()
