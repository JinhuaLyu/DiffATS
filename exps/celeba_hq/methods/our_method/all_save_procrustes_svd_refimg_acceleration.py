#!/usr/bin/env python3
"""
Accelerated Procrustes-aligned per-image channel-wise patch PCA for
CelebA-HQ 1024x1024, patch=32, rank=32.

Key accelerations vs the original script:
1. Replaces full SVD with torch.pca_lowrank (only computes a low-rank approximation).
2. Keeps all linear algebra on one device (GPU if available).
3. Avoids unnecessary synchronization and extra copies.
4. Uses lightweight sanity checks instead of expensive debug work.

For PATCH=32 on 1024x1024 images:
- num_patches = (1024 / 32)^2 = 1024
- patch_dim   = 32 * 32 = 1024
- alpha       = (1024, 32)
- V_hat       = (1024, 32)

Output shard format:
{
    "alpha":     (B, 3, 1024, 32),
    "V_hat":     (B, 3, 1024, 32),
    "filenames": [...],
    "patch":     32,
    "rank":      32,
}

Reference anchor format:
ref_anchor.pt -> {
    "D_ref":        (3, 1024, 32),
    "mean_ref":     (3, 1024),
    "ref_filename": str,
}
"""

from __future__ import annotations

import glob
import os
import random
import sys
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
IMG_DIR = "/anvil/projects/x-eng260004/factor_diffusion/original_data/celeba"
IMG_SIZE = 1024
PATCH = 32
RANK = 32
SHARD_SIZE = 500
REF_SEED = 42

# Low-rank PCA parameters
PCA_OVERSAMPLE = 8  # extra dimensions beyond rank
PCA_NITER = 2       # power iterations; 1-2 is usually enough here

OUT_DIR = "/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/celeba/our_method"
os.makedirs(OUT_DIR, exist_ok=True)

# Speed-friendly backend settings
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)


# -----------------------------
# I/O
# -----------------------------
def load_rgb(path: str, image_size: int = IMG_SIZE) -> torch.Tensor:
    """Load image -> float32 in [0, 1], shape (3, image_size, image_size)."""
    img = Image.open(path).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    return x.permute(2, 0, 1).contiguous()


# -----------------------------
# Patch extraction
# -----------------------------
def channel_to_patch_matrix(xc: torch.Tensor, p: int = PATCH) -> torch.Tensor:
    """
    xc: (H, W) single channel
    Returns A: (N, d), where
      N = (H/p)*(W/p)
      d = p*p

    For IMG_SIZE=1024, PATCH=32:
      A.shape = (1024, 1024)
    """
    return (
        xc.unfold(0, p, p)
          .unfold(1, p, p)
          .contiguous()
          .reshape(-1, p * p)
    )


# -----------------------------
# Low-rank PCA helper
# -----------------------------
def lowrank_pca_basis(
    A_cen: torch.Tensor,
    rank: int = RANK,
    oversample: int = PCA_OVERSAMPLE,
    niter: int = PCA_NITER,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes a low-rank PCA/SVD approximation of A_cen using torch.pca_lowrank.

    Returns
    -------
    U_r : (N, rank)
    S_r : (rank,)
    V_r : (d, rank)
    """
    m, n = A_cen.shape
    q = min(rank + oversample, m, n)
    U, S, V = torch.pca_lowrank(A_cen, q=q, center=False, niter=niter)
    return U[:, :rank], S[:rank], V[:, :rank]


# -----------------------------
# Build reference anchor
# -----------------------------
def build_reference_anchor(
    path: str,
    rank: int = RANK,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds D_ref and mean_ref from a single reference image.

    Returns
    -------
    D_ref   : (3, p*p, rank)
    mean_ref: (3, p*p)
    """
    x = load_rgb(path).to(device, non_blocking=True)  # (3, 1024, 1024)

    # Channel 0 is the intra-image anchor
    A_0 = channel_to_patch_matrix(x[0], p=PATCH)      # (1024, 1024)
    mean_0 = A_0.mean(dim=0)                          # (1024,)
    U0, S0, V_0 = lowrank_pca_basis(A_0 - mean_0[None, :], rank=rank)
    del U0, S0

    D_ref_list = [V_0]
    mean_list = [mean_0]

    # Channels 1 and 2 are aligned to V_0
    for c in range(1, 3):
        A_c = channel_to_patch_matrix(x[c], p=PATCH)  # (1024, 1024)
        mean_c = A_c.mean(dim=0)
        Uc, Sc, V_c = lowrank_pca_basis(A_c - mean_c[None, :], rank=rank)
        del Uc, Sc

        M = V_c.T @ V_0
        U_p, _, Vh_p = torch.linalg.svd(M, full_matrices=False)
        Q = U_p @ Vh_p
        V_hat_c = V_c @ Q

        D_ref_list.append(V_hat_c)
        mean_list.append(mean_c)

    D_ref = torch.stack(D_ref_list, dim=0)    # (3, 1024, 16)
    mean_ref = torch.stack(mean_list, dim=0)  # (3, 1024)
    return D_ref, mean_ref


# -----------------------------
# Per-channel Procrustes PCA
# -----------------------------
def procrustes_svd_channel(
    A_c: torch.Tensor,
    D_c: torch.Tensor,
    mean_c: torch.Tensor,
    rank: int = RANK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A_c   : (N, p*p)
    D_c   : (p*p, rank)
    mean_c: (p*p,)

    Returns
    -------
    alpha : (N, rank)
    V_hat : (p*p, rank)
    """
    A_cen = A_c - mean_c[None, :]
    U_r, S_r, V_r = lowrank_pca_basis(A_cen, rank=rank)

    M = V_r.T @ D_c
    U_p, _, Vh_p = torch.linalg.svd(M, full_matrices=False)
    Q = U_p @ Vh_p

    V_hat = V_r @ Q
    alpha = (U_r * S_r[None, :]) @ Q
    return alpha, V_hat


# -----------------------------
# Save shard
# -----------------------------
def save_shard(
    idx: int,
    alpha_list: List[torch.Tensor],
    vhat_list: List[torch.Tensor],
    names: List[str],
) -> None:
    alpha_stack = torch.stack(alpha_list, dim=0)  # (B, 3, 1024, 16)
    vhat_stack = torch.stack(vhat_list, dim=0)    # (B, 3, 1024, 16)

    shard_path = os.path.join(
        OUT_DIR,
        f"celebahq{IMG_SIZE}_patchsvd_procrustes_refimg_p{PATCH}_r{RANK}_shard_{idx:04d}.pt",
    )
    torch.save(
        {
            "alpha": alpha_stack,
            "V_hat": vhat_stack,
            "filenames": names,
            "patch": PATCH,
            "rank": RANK,
        },
        shard_path,
    )


# -----------------------------
# Main
# -----------------------------
def find_resume_point(out_dir: str, shard_size: int) -> Tuple[int, int]:
    """
    Scan OUT_DIR for already-written shards. Return (next_shard_idx, n_done_imgs).
    A shard is "done" only if it loads cleanly and has the expected B == SHARD_SIZE
    (we only allow the *last* shard to be ragged, but here we resume strictly on
    SHARD_SIZE multiples to keep image-index alignment intact).
    """
    pattern = os.path.join(
        out_dir, f"celebahq{IMG_SIZE}_patchsvd_procrustes_refimg_p{PATCH}_r{RANK}_shard_*.pt"
    )
    next_idx = 0
    n_done = 0
    while True:
        expected = os.path.join(
            out_dir,
            f"celebahq{IMG_SIZE}_patchsvd_procrustes_refimg_p{PATCH}_r{RANK}_shard_{next_idx:04d}.pt",
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


def main() -> None:
    img_paths = sorted(
        glob.glob(os.path.join(IMG_DIR, "*.jpg"))
        + glob.glob(os.path.join(IMG_DIR, "*.jpeg"))
        + glob.glob(os.path.join(IMG_DIR, "*.png"))
    )
    if not img_paths:
        raise FileNotFoundError(f"No image files found in {IMG_DIR}")

    rng = random.Random(REF_SEED)
    ref_path = rng.choice(img_paths)
    ref_name = os.path.basename(ref_path)
    print(f"Reference image: {ref_name} (seed={REF_SEED})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    anchor_path = os.path.join(OUT_DIR, "ref_anchor.pt")
    if os.path.exists(anchor_path):
        anchor = torch.load(anchor_path, map_location="cpu", weights_only=False)
        D_ref    = anchor["D_ref"].to(device)
        mean_ref = anchor["mean_ref"].to(device)
        print(f"[Resume] loaded existing ref_anchor.pt (ref={anchor.get('ref_filename')})")
    else:
        D_ref, mean_ref = build_reference_anchor(ref_path, rank=RANK, device=device)
        print(f"D_ref shape: {tuple(D_ref.shape)}")
        print(f"mean_ref shape: {tuple(mean_ref.shape)}")
        torch.save(
            {"D_ref": D_ref.cpu(), "mean_ref": mean_ref.cpu(), "ref_filename": ref_name},
            anchor_path,
        )
        print(f"Anchor saved -> {anchor_path}")
        # restore device tensors after save
        D_ref    = D_ref.to(device)
        mean_ref = mean_ref.to(device)
    print(f"Saving shards to:\n  {OUT_DIR}\n")

    shard_idx, n_done = find_resume_point(OUT_DIR, SHARD_SIZE)
    if n_done > 0:
        print(f"[Resume] {n_done} images already processed in {shard_idx} shards; skipping ahead.")
    img_paths = img_paths[n_done:]
    print(f"[Resume] remaining: {len(img_paths)} images, next shard idx={shard_idx}")

    alpha_buf: List[torch.Tensor] = []
    vhat_buf: List[torch.Tensor] = []
    name_buf: List[str] = []

    for path in tqdm(img_paths, desc="Accelerated Procrustes-aligned patch PCA"):
        x = load_rgb(path).to(device, non_blocking=True)

        alpha_ch: List[torch.Tensor] = []
        vhat_ch: List[torch.Tensor] = []
        for c in range(3):
            A_c = channel_to_patch_matrix(x[c], p=PATCH)   # (1024, 1024)
            alpha_c, V_hat_c = procrustes_svd_channel(A_c, D_ref[c], mean_ref[c], rank=RANK)
            alpha_ch.append(alpha_c)
            vhat_ch.append(V_hat_c)

        alpha_img = torch.stack(alpha_ch, dim=0)  # (3, 1024, 16)
        vhat_img = torch.stack(vhat_ch, dim=0)    # (3, 1024, 16)

        alpha_buf.append(alpha_img.cpu())
        vhat_buf.append(vhat_img.cpu())
        name_buf.append(os.path.basename(path))

        if len(alpha_buf) >= SHARD_SIZE:
            save_shard(shard_idx, alpha_buf, vhat_buf, name_buf)
            shard_idx += 1
            alpha_buf, vhat_buf, name_buf = [], [], []

    if alpha_buf:
        save_shard(shard_idx, alpha_buf, vhat_buf, name_buf)

    shards = sorted(
        glob.glob(os.path.join(OUT_DIR, f"celebahq{IMG_SIZE}_patchsvd_procrustes_refimg_p{PATCH}_r{RANK}_shard_*.pt"))
    )
    with open(os.path.join(OUT_DIR, "manifest.txt"), "w") as f:
        for s in shards:
            f.write(os.path.basename(s) + "\n")

    num_patches = (IMG_SIZE // PATCH) ** 2
    patch_dim = PATCH * PATCH
    joint_len = num_patches + patch_dim

    print(f"\nDone. {len(shards)} shard(s) saved.")
    print(f"alpha: (B, 3, {num_patches}, {RANK})")
    print(f"V_hat: (B, 3, {patch_dim}, {RANK})")
    print(f"Joint: cat([alpha, V_hat], dim=2) -> (B, 3, {joint_len}, {RANK})")

    # Recount total images covered by valid shards (including any final ragged shard)
    total_in_shards = 0
    for s in shards:
        d = torch.load(s, map_location="cpu", weights_only=False)
        total_in_shards += d["alpha"].shape[0]

    all_paths = sorted(
        glob.glob(os.path.join(IMG_DIR, "*.jpg"))
        + glob.glob(os.path.join(IMG_DIR, "*.jpeg"))
        + glob.glob(os.path.join(IMG_DIR, "*.png"))
    )
    n_expected = len(all_paths)
    print(f"\nCoverage: {total_in_shards}/{n_expected} images in shards.")

    if total_in_shards != n_expected:
        print(f"[INCOMPLETE] expected {n_expected}, got {total_in_shards}. Exiting non-zero.",
              file=sys.stderr)
        sys.exit(1)

    # Lightweight sanity check on the first shard / first image
    print("\n--- Lightweight sanity check ---")
    shard = torch.load(shards[0], map_location="cpu", weights_only=False)
    alpha_s = shard["alpha"][0]
    vhat_s = shard["V_hat"][0]
    print(f"first alpha shape: {tuple(alpha_s.shape)}")
    print(f"first V_hat shape: {tuple(vhat_s.shape)}")


if __name__ == "__main__":
    main()
