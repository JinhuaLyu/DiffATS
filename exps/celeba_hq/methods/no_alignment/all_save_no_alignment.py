#!/usr/bin/env python3
"""
Per-image, per-channel patch SVD WITHOUT Procrustes alignment.

This is the ablation counterpart to all_save_procrustes_svd_refimg_acceleration.py:
each image's per-channel rank-R basis is taken straight from torch.pca_lowrank
with NO rotation toward a reference / global anchor. As a consequence,
V_hat across images can differ by an arbitrary orthogonal RxR rotation
(and arbitrary sign per column). alpha is the matching unrotated coefficient
matrix.

For PATCH=32 on 1024x1024:
- num_patches = (1024/32)^2 = 1024
- patch_dim   = 32*32       = 1024
- alpha       = (1024, 32)
- V_hat       = (1024, 32)

Output shard format (B = SHARD_SIZE):
  {
    "alpha":     (B, 3, 1024, 32),
    "V_hat":     (B, 3, 1024, 32),
    "filenames": [...],
    "patch":     32,
    "rank":      32,
  }

NOTE: there is no ref_anchor.pt because the dataset has no shared anchor.
The per-channel mean used to center each patch matrix IS saved per image
inside the shard ("mean_per_image") so reconstructions stay exact.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ----------------- Config -----------------
IMG_DIR = "/anvil/projects/x-eng260004/factor_diffusion/original_data/celeba"
OUT_DIR = "/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/celeba/no_alignment"

IMG_SIZE   = 1024
PATCH      = 32
RANK       = 32
SHARD_SIZE = 500

PCA_OVERSAMPLE = 8
PCA_NITER      = 2

os.makedirs(OUT_DIR, exist_ok=True)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)


# ----------------- I/O -----------------
def load_rgb(path: str, image_size: int = IMG_SIZE) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    return x.permute(2, 0, 1).contiguous()


def channel_to_patch_matrix(xc: torch.Tensor, p: int = PATCH) -> torch.Tensor:
    """xc: (H, W) -> (N, d) with N = (H/p)*(W/p), d = p*p."""
    return (
        xc.unfold(0, p, p)
          .unfold(1, p, p)
          .contiguous()
          .reshape(-1, p * p)
    )


# ----------------- Low-rank PCA helper -----------------
def lowrank_pca(
    A_cen: torch.Tensor,
    rank: int = RANK,
    oversample: int = PCA_OVERSAMPLE,
    niter: int = PCA_NITER,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (U_r:(N,rank), S_r:(rank,), V_r:(d,rank)) via randomized SVD."""
    m, n = A_cen.shape
    q = min(rank + oversample, m, n)
    U, S, V = torch.pca_lowrank(A_cen, q=q, center=False, niter=niter)
    return U[:, :rank], S[:rank], V[:, :rank]


# ----------------- Per-channel SVD (no alignment) -----------------
def patch_svd_channel_no_align(
    A_c: torch.Tensor,
    rank: int = RANK,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A_c : (N, d)
    returns
      alpha  : (N, rank)        = U_r @ diag(S_r)
      V_hat  : (d, rank)        = V_r        (no rotation)
      mean_c : (d,)
    """
    mean_c = A_c.mean(dim=0)
    U_r, S_r, V_r = lowrank_pca(A_c - mean_c[None, :], rank=rank)
    alpha = U_r * S_r[None, :]
    V_hat = V_r
    return alpha, V_hat, mean_c


# ----------------- Save shard -----------------
def save_shard(
    idx: int,
    alpha_list: List[torch.Tensor],
    vhat_list: List[torch.Tensor],
    mean_list: List[torch.Tensor],
    names: List[str],
) -> None:
    alpha_stack = torch.stack(alpha_list, dim=0)  # (B, 3, N, rank)
    vhat_stack  = torch.stack(vhat_list,  dim=0)  # (B, 3, d, rank)
    mean_stack  = torch.stack(mean_list,  dim=0)  # (B, 3, d)

    shard_path = os.path.join(
        OUT_DIR,
        f"celebahq{IMG_SIZE}_patchsvd_no_alignment_p{PATCH}_r{RANK}_shard_{idx:04d}.pt",
    )
    torch.save(
        {
            "alpha":          alpha_stack,
            "V_hat":          vhat_stack,
            "mean_per_image": mean_stack,
            "filenames":      names,
            "patch":          PATCH,
            "rank":           RANK,
        },
        shard_path,
    )


# ----------------- Resume -----------------
def find_resume_point(out_dir: str, shard_size: int) -> Tuple[int, int]:
    """Return (next_shard_idx, n_done_imgs) using only contiguous full SHARD_SIZE shards."""
    next_idx = 0
    n_done = 0
    while True:
        expected = os.path.join(
            out_dir,
            f"celebahq{IMG_SIZE}_patchsvd_no_alignment_p{PATCH}_r{RANK}_shard_{next_idx:04d}.pt",
        )
        if not os.path.exists(expected):
            break
        try:
            data = torch.load(expected, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[Resume] {os.path.basename(expected)} load failed: {e}; will redo from {next_idx}")
            break
        B = data["alpha"].shape[0]
        if B != shard_size:
            print(f"[Resume] {os.path.basename(expected)} has B={B} != {shard_size}; will redo from {next_idx}")
            break
        n_done  += B
        next_idx += 1
    return next_idx, n_done


# ----------------- Main -----------------
def main() -> None:
    img_paths_all = sorted(
        glob.glob(os.path.join(IMG_DIR, "*.png"))
        + glob.glob(os.path.join(IMG_DIR, "*.jpg"))
        + glob.glob(os.path.join(IMG_DIR, "*.jpeg"))
    )
    if not img_paths_all:
        raise FileNotFoundError(f"No images in {IMG_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] images = {len(img_paths_all)}")
    print(f"[INFO] PATCH={PATCH}  RANK={RANK}  SHARD_SIZE={SHARD_SIZE}")
    print(f"[INFO] OUT_DIR = {OUT_DIR}")

    shard_idx, n_done = find_resume_point(OUT_DIR, SHARD_SIZE)
    if n_done > 0:
        print(f"[Resume] {n_done} images already in {shard_idx} shards; continuing.")
    img_paths = img_paths_all[n_done:]
    print(f"[Resume] remaining: {len(img_paths)}, next shard idx={shard_idx}")

    alpha_buf: List[torch.Tensor] = []
    vhat_buf:  List[torch.Tensor] = []
    mean_buf:  List[torch.Tensor] = []
    name_buf:  List[str]          = []

    for path in tqdm(img_paths, desc="Per-image patch SVD (no alignment)"):
        x = load_rgb(path).to(device, non_blocking=True)

        alpha_ch, vhat_ch, mean_ch = [], [], []
        for c in range(3):
            A_c = channel_to_patch_matrix(x[c], p=PATCH)             # (N, d)
            alpha_c, V_hat_c, mean_c = patch_svd_channel_no_align(A_c, rank=RANK)
            alpha_ch.append(alpha_c)
            vhat_ch.append(V_hat_c)
            mean_ch.append(mean_c)

        alpha_img = torch.stack(alpha_ch, dim=0).cpu()   # (3, N, RANK)
        vhat_img  = torch.stack(vhat_ch,  dim=0).cpu()   # (3, d, RANK)
        mean_img  = torch.stack(mean_ch,  dim=0).cpu()   # (3, d)

        alpha_buf.append(alpha_img)
        vhat_buf.append(vhat_img)
        mean_buf.append(mean_img)
        name_buf.append(os.path.basename(path))

        if len(alpha_buf) >= SHARD_SIZE:
            save_shard(shard_idx, alpha_buf, vhat_buf, mean_buf, name_buf)
            shard_idx += 1
            alpha_buf, vhat_buf, mean_buf, name_buf = [], [], [], []

    if alpha_buf:
        save_shard(shard_idx, alpha_buf, vhat_buf, mean_buf, name_buf)
        shard_idx += 1

    shards = sorted(glob.glob(
        os.path.join(OUT_DIR, f"celebahq{IMG_SIZE}_patchsvd_no_alignment_p{PATCH}_r{RANK}_shard_*.pt")
    ))
    with open(os.path.join(OUT_DIR, "manifest.txt"), "w") as f:
        for s in shards:
            f.write(os.path.basename(s) + "\n")

    total_in_shards = 0
    for s in shards:
        d = torch.load(s, map_location="cpu", weights_only=False)
        total_in_shards += d["alpha"].shape[0]
    n_expected = len(img_paths_all)
    print(f"\nCoverage: {total_in_shards}/{n_expected} images.")

    if shards:
        sample = torch.load(shards[0], map_location="cpu", weights_only=False)
        print(f"[CHECK] first shard: alpha {tuple(sample['alpha'].shape)}, "
              f"V_hat {tuple(sample['V_hat'].shape)}, "
              f"mean_per_image {tuple(sample['mean_per_image'].shape)}, "
              f"n_filenames={len(sample['filenames'])}")

    print(f"\n[DONE] {len(shards)} shards in {OUT_DIR}")

    if total_in_shards != n_expected:
        print(f"[INCOMPLETE] expected {n_expected}, got {total_in_shards}. Exiting non-zero.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
