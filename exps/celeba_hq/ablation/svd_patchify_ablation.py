"""
Ablation: SVD reconstruction with vs. without 32x32 patchification on CelebA-HQ.

For each of N randomly-sampled 1024x1024 images, build two per-channel matrices:
  (A) patch:   img -> (1024 patches, 1024 px)  via 32x32 patchification, then unpatchify back
  (B) raw:     img -> (1024, 1024) directly
SVD low-rank approximations are computed for ranks in RANKS, then RMSE / PSNR /
relative Frobenius error are reported per image and averaged.
"""

import os
import gc
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR  = "${DATA_ROOT}/bkim/CelebA-HQ/celeba_hq_images/all"
OUT_DIR   = "${REPO_ROOT}/exps/celeba_hq/ablation"
os.makedirs(OUT_DIR, exist_ok=True)

PATCH_SIZE = 32
RANKS      = [16, 32, 64, 128]
N_IMAGES   = int(os.environ.get("N_IMAGES", "100"))
SEED       = 42
MAX_PIXEL  = 255.0
MAKE_FIGURE = N_IMAGES <= 20


def patchify(img_c: np.ndarray, p: int) -> np.ndarray:
    H, W = img_c.shape
    nH, nW = H // p, W // p
    return (img_c
            .reshape(nH, p, nW, p)
            .transpose(0, 2, 1, 3)
            .reshape(nH * nW, p * p))


def unpatchify(M: np.ndarray, p: int, H: int, W: int) -> np.ndarray:
    nH, nW = H // p, W // p
    return (M
            .reshape(nH, nW, p, p)
            .transpose(0, 2, 1, 3)
            .reshape(H, W))


def psnr_from_rmse(rmse: float, max_val: float = MAX_PIXEL) -> float:
    if rmse == 0:
        return float("inf")
    return 20.0 * np.log10(max_val / rmse)


def metrics_from_singular_values(s: np.ndarray, M_shape) -> dict:
    """Closed-form RMSE / RelErr for rank-r truncation given singular values."""
    m, n = M_shape
    s64 = s.astype(np.float64)
    full_sq = float(np.sum(s64 ** 2))
    out = {}
    for r in RANKS:
        tail_sq = float(np.sum(s64[r:] ** 2))
        rmse = (tail_sq / (m * n)) ** 0.5
        rel  = (tail_sq / full_sq) ** 0.5 if full_sq > 0 else 0.0
        out[r] = {"rmse": rmse, "rel_err": rel}
    return out


def run_variant_metrics_only(img: np.ndarray, use_patch: bool):
    """Channel-wise SVD metrics from singular values only (no U/V/recon).
       Returns: dict rank -> {"rmse", "rel_err", "psnr"} averaged over channels."""
    H, W, C = img.shape
    chan = {r: {"rmse": [], "rel_err": []} for r in RANKS}
    for c in range(C):
        M = patchify(img[:, :, c], PATCH_SIZE) if use_patch else img[:, :, c]
        s = np.linalg.svd(M, compute_uv=False)
        m_per = metrics_from_singular_values(s, M.shape)
        for r in RANKS:
            chan[r]["rmse"].append(m_per[r]["rmse"])
            chan[r]["rel_err"].append(m_per[r]["rel_err"])
        del s, M
    out = {}
    for r in RANKS:
        rmse = float(np.mean(chan[r]["rmse"]))
        rel  = float(np.mean(chan[r]["rel_err"]))
        out[r] = {"rmse": rmse, "rel_err": rel, "psnr": psnr_from_rmse(rmse)}
    return out


def run_variant(matrices: np.ndarray, img_shape, use_patch: bool):
    """matrices: (C, m, n).  Returns (results_per_rank, recon_per_rank)."""
    H, W, C = img_shape
    USVts = [np.linalg.svd(matrices[c], full_matrices=False) for c in range(C)]
    results, recs = {}, {}
    for rank in RANKS:
        rmse_l, rel_l, ch_recs = [], [], []
        for c in range(C):
            U, s, Vt = USVts[c]
            M_c = matrices[c]
            M_r = U[:, :rank] @ (s[:rank, None] * Vt[:rank, :])
            diff = M_c - M_r
            rmse_l.append(float(np.sqrt(np.mean(diff ** 2))))
            rel_l.append(float(np.linalg.norm(diff, "fro") / np.linalg.norm(M_c, "fro")))
            ch_recs.append(unpatchify(M_r, PATCH_SIZE, H, W) if use_patch else M_r)
        rec_img = np.clip(np.stack(ch_recs, axis=-1), 0, 255).astype(np.uint8)
        recs[rank] = rec_img
        mean_rmse = float(np.mean(rmse_l))
        results[rank] = {
            "rmse":    mean_rmse,
            "rel_err": float(np.mean(rel_l)),
            "psnr":    psnr_from_rmse(mean_rmse),
        }
    return results, recs


def main():
    rng = np.random.default_rng(SEED)
    all_files = sorted(f for f in os.listdir(DATA_DIR)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"Total images available: {len(all_files)}")
    chosen_idx = sorted(rng.choice(len(all_files), size=N_IMAGES, replace=False).tolist())
    chosen_files = [all_files[i] for i in chosen_idx]
    print(f"Selected (seed={SEED}): {chosen_files}")

    per_img_patch, per_img_raw = [], []
    recs_patch_all, recs_raw_all, originals = [], [], []

    for ip, fname in enumerate(chosen_files):
        path = os.path.join(DATA_DIR, fname)
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        H, W, C = img.shape
        assert (H, W) == (1024, 1024), f"{fname} has shape {img.shape}"

        if MAKE_FIGURE:
            patch_mats = np.stack([patchify(img[:, :, c], PATCH_SIZE) for c in range(C)], axis=0)
            raw_mats   = np.stack([img[:, :, c] for c in range(C)], axis=0)
            rp, recp = run_variant(patch_mats, (H, W, C), use_patch=True)
            rr, recr = run_variant(raw_mats,   (H, W, C), use_patch=False)
            recs_patch_all.append(recp); recs_raw_all.append(recr)
            originals.append(img.astype(np.uint8))
            del patch_mats, raw_mats
        else:
            rp = run_variant_metrics_only(img, use_patch=True)
            rr = run_variant_metrics_only(img, use_patch=False)

        per_img_patch.append(rp); per_img_raw.append(rr)
        del img
        gc.collect()
        print(f"  [{ip + 1}/{N_IMAGES}] {fname}", flush=True)

    avg_patch = {r: {k: float(np.mean([per_img_patch[i][r][k] for i in range(N_IMAGES)]))
                     for k in ("rmse", "rel_err", "psnr")} for r in RANKS}
    avg_raw   = {r: {k: float(np.mean([per_img_raw[i][r][k]   for i in range(N_IMAGES)]))
                     for k in ("rmse", "rel_err", "psnr")} for r in RANKS}

    # -- metrics: per-image rows + AVG row, both variants --
    txt_path = os.path.join(OUT_DIR, "metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Ablation: SVD with vs. without {PATCH_SIZE}x{PATCH_SIZE} patchification\n")
        f.write(f"Sampled {N_IMAGES} images (seed={SEED}) from {DATA_DIR}\n")
        f.write(f"Selected files: {chosen_files}\n\n")
        for name, per_img, avg in [
            ("WITH patchification", per_img_patch, avg_patch),
            ("WITHOUT patchification (raw 1024x1024)", per_img_raw, avg_raw),
        ]:
            f.write(f"-- {name} --\n")
            hdr = f"{'image':<12}" + "".join(
                f"{'r' + str(r) + ' RMSE':>11}{'r' + str(r) + ' PSNR':>11}{'r' + str(r) + ' Rel':>11}"
                for r in RANKS) + "\n"
            f.write(hdr)
            f.write("-" * len(hdr) + "\n")
            for i in range(N_IMAGES):
                row = f"{chosen_files[i]:<12}"
                for r in RANKS:
                    pi = per_img[i][r]
                    row += f"{pi['rmse']:>11.4f}{pi['psnr']:>11.4f}{pi['rel_err']:>11.6f}"
                f.write(row + "\n")
            f.write("-" * len(hdr) + "\n")
            row = f"{'AVG':<12}"
            for r in RANKS:
                row += f"{avg[r]['rmse']:>11.4f}{avg[r]['psnr']:>11.4f}{avg[r]['rel_err']:>11.6f}"
            f.write(row + "\n\n")

        f.write("-- DELTA (no-patch - patch). PSNR delta is no-patch - patch (negative => patch better).\n")
        f.write("    RMSE/Rel delta is no-patch - patch (positive => patch better).\n")
        hdr = f"{'metric':<12}" + "".join(f"{'r' + str(r):>14}" for r in RANKS) + "\n"
        f.write(hdr); f.write("-" * len(hdr) + "\n")
        for k, lbl in [("rmse", "DeltaRMSE"), ("psnr", "DeltaPSNR"), ("rel_err", "DeltaRelErr")]:
            row = f"{lbl:<12}"
            for r in RANKS:
                row += f"{avg_raw[r][k] - avg_patch[r][k]:>14.4f}"
            f.write(row + "\n")
    print(f"Metrics text -> {txt_path}")

    # CSV
    csv_path = os.path.join(OUT_DIR, "metrics.csv")
    with open(csv_path, "w") as f:
        cols = ["variant", "image"]
        for r in RANKS:
            cols += [f"rank{r}_RMSE", f"rank{r}_PSNR", f"rank{r}_RelErr"]
        f.write(",".join(cols) + "\n")
        for variant, per_img, avg in [
            ("patch", per_img_patch, avg_patch),
            ("raw",   per_img_raw,   avg_raw),
        ]:
            for i in range(N_IMAGES):
                row = [variant, chosen_files[i]]
                for r in RANKS:
                    pi = per_img[i][r]
                    row += [f"{pi['rmse']:.6f}", f"{pi['psnr']:.6f}", f"{pi['rel_err']:.6f}"]
                f.write(",".join(row) + "\n")
            row = [variant, "AVG"]
            for r in RANKS:
                row += [f"{avg[r]['rmse']:.6f}", f"{avg[r]['psnr']:.6f}", f"{avg[r]['rel_err']:.6f}"]
            f.write(",".join(row) + "\n")
    print(f"Metrics CSV  -> {csv_path}")

    if not MAKE_FIGURE:
        print("\n-- Summary (averaged over images) --")
        print(f"{'rank':>6}  {'patch RMSE':>12}{'patch PSNR':>12}{'raw RMSE':>12}{'raw PSNR':>12}")
        for r in RANKS:
            print(f"{r:>6}  "
                  f"{avg_patch[r]['rmse']:>12.4f}{avg_patch[r]['psnr']:>12.4f}"
                  f"{avg_raw[r]['rmse']:>12.4f}{avg_raw[r]['psnr']:>12.4f}")
        return

    # -- overview figure: 2*N_IMAGES rows x (1 + |RANKS|) cols --
    nrows = 2 * N_IMAGES
    ncols = 1 + len(RANKS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6))
    fig.suptitle(
        f"CelebA-HQ SVD Ablation -- Patch ({PATCH_SIZE}x{PATCH_SIZE}) vs. No-patch  "
        f"({N_IMAGES} images, seed={SEED})",
        fontsize=14, y=0.999,
    )
    for i in range(N_IMAGES):
        for row_off, label, per_img, recs in [
            (0, "patch",    per_img_patch, recs_patch_all[i]),
            (1, "no-patch", per_img_raw,   recs_raw_all[i]),
        ]:
            r_idx = 2 * i + row_off
            axes[r_idx, 0].imshow(originals[i])
            ttl = f"{chosen_files[i]}\n[{label}] original" if row_off == 0 else f"[{label}] original"
            axes[r_idx, 0].set_title(ttl, fontsize=9)
            axes[r_idx, 0].axis("off")
            for col, r in enumerate(RANKS, start=1):
                m = per_img[i][r]
                axes[r_idx, col].imshow(recs[r])
                axes[r_idx, col].set_title(
                    f"{label}  rank={r}\nRMSE={m['rmse']:.2f}  PSNR={m['psnr']:.2f}\nRelErr={m['rel_err']:.4f}",
                    fontsize=8,
                )
                axes[r_idx, col].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.995))
    fig_path = os.path.join(OUT_DIR, "overview.png")
    plt.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Overview     -> {fig_path}")

    # console summary
    print("\n-- Summary (averaged over images) --")
    print(f"{'rank':>6}  {'patch RMSE':>12}{'patch PSNR':>12}{'raw RMSE':>12}{'raw PSNR':>12}")
    for r in RANKS:
        print(f"{r:>6}  "
              f"{avg_patch[r]['rmse']:>12.4f}{avg_patch[r]['psnr']:>12.4f}"
              f"{avg_raw[r]['rmse']:>12.4f}{avg_raw[r]['psnr']:>12.4f}")


if __name__ == "__main__":
    main()
