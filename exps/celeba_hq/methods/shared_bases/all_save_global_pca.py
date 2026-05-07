#!/usr/bin/env python3
"""
Per-channel GLOBAL (shared) PCA over all CelebA-HQ 1024x1024 images.

Pipeline:
  Step 1 (one I/O pass):
    For each channel c independently, accumulate count, sum, and outer-sum
    over all 32x32 patches across all 30,000 images.
        count[c]      += N_patches per image
        sum[c]        += sum of patches            (d,)
        sum_outer[c]  += A_c.T @ A_c               (d, d)
  Step 2 (closed form):
    mean[c]  = sum[c] / count[c]
    cov[c]   = sum_outer[c] / count[c] - mean[c] mean[c].T
    eigh(cov[c]) -> top RANK eigenvectors -> V[c] (d, RANK)
  Step 3 (second I/O pass):
    For each image, each channel c:
        alpha[c] = (A_c - mean[c]) @ V[c]    # (N, RANK)
    Save in shards of SHARD_SIZE images each.

Output layout
-------------
{OUT_DIR}/
  global_dict.pt:
    {
      "D":                       (3, d, RANK)   # the global per-channel basis
      "mean":                    (3, d)         # per-channel patch mean
      "explained_variance":      (3, RANK)
      "explained_variance_ratio":(3, RANK)
      "patch_count_per_channel": (3,)           # int
      "patch":                   PATCH
      "rank":                    RANK
    }
  celebahq1024_global_pca_p32_r32_shard_XXXX.pt:
    {
      "alpha":     (B, 3, N, RANK)   # = (B, 3, 1024, 32)
      "filenames": [str, ...]
      "patch":     PATCH
      "rank":      RANK
    }
  manifest.txt          # one shard filename per line
"""

from __future__ import annotations

import glob
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ----------------- Config -----------------
IMG_DIR = "${DATA_ROOT}/original_data/celeba"
OUT_DIR = "${DATA_ROOT}/tucker_factors/celeba/shared_bases"

IMG_SIZE   = 1024
PATCH      = 32
RANK       = 32
SHARD_SIZE = 500

PATCH_DIM = PATCH * PATCH                    # 1024
N_PATCH   = (IMG_SIZE // PATCH) ** 2         # 1024 patches per channel per image

os.makedirs(OUT_DIR, exist_ok=True)

# Backend speed flags
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)


# ----------------- I/O -----------------
def load_rgb(path: str, image_size: int = IMG_SIZE) -> torch.Tensor:
    """Load image -> float32 in [0, 1], shape (3, image_size, image_size)."""
    img = Image.open(path).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    return x.permute(2, 0, 1).contiguous()


def channel_to_patch_matrix(xc: torch.Tensor, p: int = PATCH) -> torch.Tensor:
    """xc: (H, W) -> A: (N, d) with N = (H/p)*(W/p), d = p*p."""
    return (
        xc.unfold(0, p, p)
          .unfold(1, p, p)
          .contiguous()
          .reshape(-1, p * p)
    )


# ----------------- Pass 1: covariance accumulation -----------------
def pass1_accumulate(img_paths: List[str], device: torch.device):
    """
    Stream all images once, accumulate per-channel:
      count[c]:     int
      sum[c]:       (d,)
      sum_outer[c]: (d, d)
    All in float64 on `device` (CPU is fine; GPU is ~10x faster for the matmul).
    """
    C, d = 3, PATCH_DIM
    count     = torch.zeros(C, dtype=torch.float64, device=device)
    sum_x     = torch.zeros(C, d, dtype=torch.float64, device=device)
    sum_outer = torch.zeros(C, d, d, dtype=torch.float64, device=device)

    for path in tqdm(img_paths, desc="Pass 1 (covariance)"):
        x = load_rgb(path).to(device, non_blocking=True)        # (3, H, W) float32
        for c in range(C):
            A = channel_to_patch_matrix(x[c], p=PATCH).double() # (N, d)
            count[c]     += A.shape[0]
            sum_x[c]     += A.sum(dim=0)
            sum_outer[c] += A.T @ A

    return count.cpu(), sum_x.cpu(), sum_outer.cpu()


# ----------------- Pass 1.5: closed-form PCA -----------------
def compute_global_pca(count, sum_x, sum_outer):
    """
    Returns
    -------
    D           : (3, d, RANK)         -- top RANK eigenvectors per channel
    mean        : (3, d)
    eigvals_top : (3, RANK)
    explained_variance_ratio : (3, RANK)
    """
    C, d = sum_x.shape

    mean = sum_x / count[:, None]                          # (3, d)
    second_moment = sum_outer / count[:, None, None]       # (3, d, d)
    cov = second_moment - mean.unsqueeze(-1) @ mean.unsqueeze(-2)  # (3, d, d)

    # Symmetrize defensively against fp drift
    cov = 0.5 * (cov + cov.transpose(-1, -2))

    D_list, eigvals_list, evr_list = [], [], []
    for c in range(C):
        # eigh returns eigvals ascending; flip to descending
        eigvals, eigvecs = torch.linalg.eigh(cov[c])               # eigvals: (d,), eigvecs: (d, d)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        eigvals_top = eigvals[:RANK].clamp_min(0.0)                # numerical floor
        D_c         = eigvecs[:, :RANK]                            # (d, RANK)

        total_var   = eigvals.clamp_min(0.0).sum()
        evr         = eigvals_top / total_var.clamp_min(1e-30)

        D_list.append(D_c)
        eigvals_list.append(eigvals_top)
        evr_list.append(evr)

        print(
            f"  channel {c}: top-{RANK} eigvals "
            f"[{eigvals_top[0].item():.6e} ... {eigvals_top[-1].item():.6e}]  "
            f"cum-explained = {evr.sum().item()*100:.2f}%"
        )

    D    = torch.stack(D_list, dim=0).float()             # (3, d, RANK)
    mean = mean.float()                                   # (3, d)
    eigvals_top = torch.stack(eigvals_list, dim=0).float()
    evr         = torch.stack(evr_list, dim=0).float()
    return D, mean, eigvals_top, evr


# ----------------- Pass 2: project -----------------
def save_shard(idx, alpha_list, names):
    alpha_stack = torch.stack(alpha_list, dim=0)          # (B, 3, N, RANK)
    shard_path = os.path.join(
        OUT_DIR,
        f"celebahq{IMG_SIZE}_global_pca_p{PATCH}_r{RANK}_shard_{idx:04d}.pt",
    )
    torch.save(
        {
            "alpha":     alpha_stack,
            "filenames": names,
            "patch":     PATCH,
            "rank":      RANK,
        },
        shard_path,
    )


def find_resume_point_pass2(out_dir: str, shard_size: int) -> Tuple[int, int]:
    """Return (next_shard_idx, n_done_imgs) using only contiguous full SHARD_SIZE shards."""
    next_idx = 0
    n_done = 0
    while True:
        expected = os.path.join(
            out_dir,
            f"celebahq{IMG_SIZE}_global_pca_p{PATCH}_r{RANK}_shard_{next_idx:04d}.pt",
        )
        if not os.path.exists(expected):
            break
        try:
            data = torch.load(expected, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[Resume] {os.path.basename(expected)} load failed: {e}; redo from {next_idx}")
            break
        B = data["alpha"].shape[0]
        if B != shard_size:
            print(f"[Resume] {os.path.basename(expected)} has B={B} != {shard_size}; redo from {next_idx}")
            break
        n_done  += B
        next_idx += 1
    return next_idx, n_done


def pass2_project(img_paths, D, mean, device, shard_idx_start: int = 0):
    """
    For each image, project each channel onto its global basis and stash alpha.
    """
    D    = D.to(device)        # (3, d, RANK), float32
    mean = mean.to(device)     # (3, d),       float32
    C    = 3

    alpha_buf: List[torch.Tensor] = []
    name_buf:  List[str]          = []
    shard_idx = shard_idx_start

    for path in tqdm(img_paths, desc="Pass 2 (project)"):
        x = load_rgb(path).to(device, non_blocking=True)
        alpha_ch = []
        for c in range(C):
            A = channel_to_patch_matrix(x[c], p=PATCH)           # (N, d) float32
            A_cen = A - mean[c][None, :]                         # (N, d)
            alpha_c = A_cen @ D[c]                               # (N, RANK)
            alpha_ch.append(alpha_c)
        alpha_img = torch.stack(alpha_ch, dim=0).cpu()           # (3, N, RANK)
        alpha_buf.append(alpha_img)
        name_buf.append(os.path.basename(path))

        if len(alpha_buf) >= SHARD_SIZE:
            save_shard(shard_idx, alpha_buf, name_buf)
            shard_idx += 1
            alpha_buf, name_buf = [], []

    if alpha_buf:
        save_shard(shard_idx, alpha_buf, name_buf)
        shard_idx += 1

    return shard_idx  # next shard idx (== total shards written from idx 0)


# ----------------- Main -----------------
def main():
    img_paths = sorted(
        glob.glob(os.path.join(IMG_DIR, "*.png"))
        + glob.glob(os.path.join(IMG_DIR, "*.jpg"))
        + glob.glob(os.path.join(IMG_DIR, "*.jpeg"))
    )
    if not img_paths:
        raise FileNotFoundError(f"No images in {IMG_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] images = {len(img_paths)}")
    print(f"[INFO] PATCH={PATCH}  RANK={RANK}  PATCH_DIM={PATCH_DIM}  N_PATCH={N_PATCH}")
    print(f"[INFO] OUT_DIR = {OUT_DIR}")

    dict_path = os.path.join(OUT_DIR, "global_dict.pt")
    expected_count = len(img_paths) * N_PATCH

    # ----- Pass 1 (skip if dict already exists) -----
    if os.path.exists(dict_path):
        print(f"[Resume] global_dict.pt found at {dict_path}; skipping pass 1.")
        dict_obj = torch.load(dict_path, map_location="cpu", weights_only=False)
        D    = dict_obj["D"]
        mean = dict_obj["mean"]
        t_pass1 = 0.0
    else:
        t0 = time.time()
        count, sum_x, sum_outer = pass1_accumulate(img_paths, device)
        t_pass1 = time.time() - t0
        print(f"[Pass 1] done in {t_pass1/60:.1f} min")
        print(f"[Pass 1] patch counts per channel: {count.long().tolist()} (expected {expected_count})")
        assert int(count[0].item()) == expected_count, "patch count mismatch"

        print("[PCA] solving per-channel eigendecomposition ...")
        D, mean, eigvals_top, evr = compute_global_pca(count, sum_x, sum_outer)

        torch.save(
            {
                "D":                          D,
                "mean":                       mean,
                "explained_variance":         eigvals_top,
                "explained_variance_ratio":   evr,
                "patch_count_per_channel":    count.long(),
                "patch":                      PATCH,
                "rank":                       RANK,
                "image_dir":                  IMG_DIR,
                "n_images":                   len(img_paths),
            },
            dict_path,
        )
        print(f"[PCA] saved global dict -> {dict_path}")
    print(f"[PCA] D shape: {tuple(D.shape)}, mean shape: {tuple(mean.shape)}")

    # ----- Pass 2 (resume from already-projected shards) -----
    shard_idx_start, n_done = find_resume_point_pass2(OUT_DIR, SHARD_SIZE)
    if n_done > 0:
        print(f"[Resume Pass 2] {n_done} images already projected in {shard_idx_start} shards; continuing.")
    img_paths_remaining = img_paths[n_done:]

    t0 = time.time()
    n_shards = pass2_project(img_paths_remaining, D, mean, device,
                             shard_idx_start=shard_idx_start)
    t_pass2 = time.time() - t0
    print(f"[Pass 2] done in {t_pass2/60:.1f} min, total shards now {n_shards}")

    # ----- Manifest + sanity check -----
    shards = sorted(glob.glob(
        os.path.join(OUT_DIR, f"celebahq{IMG_SIZE}_global_pca_p{PATCH}_r{RANK}_shard_*.pt")
    ))
    with open(os.path.join(OUT_DIR, "manifest.txt"), "w") as f:
        for s in shards:
            f.write(os.path.basename(s) + "\n")

    total_in_shards = 0
    for s in shards:
        d = torch.load(s, map_location="cpu", weights_only=False)
        total_in_shards += d["alpha"].shape[0]
    print(f"\nCoverage: {total_in_shards}/{len(img_paths)} images.")

    if shards:
        sample = torch.load(shards[0], map_location="cpu", weights_only=False)
        print(f"[CHECK] first shard: alpha {tuple(sample['alpha'].shape)}, "
              f"n_filenames={len(sample['filenames'])}")

    print(f"\n[DONE] {len(shards)} shards in {OUT_DIR}")
    print(f"  Pass 1 (covariance): {t_pass1/60:.1f} min")
    print(f"  Pass 2 (project)   : {t_pass2/60:.1f} min")
    print(f"  Total              : {(t_pass1+t_pass2)/60:.1f} min")

    if total_in_shards != len(img_paths):
        print(f"[INCOMPLETE] expected {len(img_paths)}, got {total_in_shards}. Exiting non-zero.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
