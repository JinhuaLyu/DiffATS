"""viz_recon.py — reconstruct (1024,200) trajectory and (1024,20) IC from
factorized tensors and compare against ground truth.

Usage:
    python viz_recon.py --idx 0 --split train
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

# Avoid headless display issues
mpl.use("Agg")

PATCH_X = 32
PATCH_T = 20
NX = 1024
T_TRAJ = 200
N_BLOCK_X = NX // PATCH_X        # 32
N_BLOCK_T = T_TRAJ // PATCH_T    # 10
IC_CHUNK = 32

ORIG_DIR = "/work/hdd/bgxp/factor_diffusion/original_data/burgers_1d"
FACTOR_DIR = "/work/hdd/bgxp/factor_diffusion/tucker_factors/burgers_1d"
DEFAULT_OUT_DIR = "/u/jlyu5/factor_diffusion/1d_physics/1d_burgers/test"


def reconstruct_traj_from_factors(alpha: torch.Tensor, V_hat: torch.Tensor) -> np.ndarray:
    """alpha (320, 32), V_hat (640, 32) -> (1024, 200) numpy.

    Row-major patch order on input: outer = N_BLOCK_X (spatial), inner = N_BLOCK_T (time).
    Inside each patch: outer = PATCH_X (spatial), inner = PATCH_T (time).
    """
    A = (alpha @ V_hat.T).cpu().numpy()                     # (320, 640)
    A = A.reshape(N_BLOCK_X, N_BLOCK_T, PATCH_X, PATCH_T)   # (32, 10, 32, 20)
    A = A.transpose(0, 2, 1, 3)                              # (32, 32, 10, 20)
    return A.reshape(N_BLOCK_X * PATCH_X, N_BLOCK_T * PATCH_T)  # (1024, 200)


def reconstruct_ic_from_factors(alpha_ic: torch.Tensor, V_hat_ic: torch.Tensor) -> np.ndarray:
    """alpha_ic (32, 32), V_hat_ic (640, 32) -> (1024, 20) numpy.

    A_ic = alpha_ic @ V_hat_ic.T -> (32, 640). Each row is a 32-element chunk
    replicated PATCH_T=20 times along time, flattened row-major (spatial slow,
    time fast). So (32, 640) -> (32, 32, 20) -> (1024, 20).
    """
    A_ic = (alpha_ic @ V_hat_ic.T).cpu().numpy()             # (32, 640)
    A_ic = A_ic.reshape(IC_CHUNK, PATCH_X, PATCH_T)          # (32 chunks, 32 spatial, 20 time)
    return A_ic.reshape(IC_CHUNK * PATCH_X, PATCH_T)         # (1024, 20)


def plot_traj_compare(orig: np.ndarray, recon: np.ndarray, out_path: Path, title_suffix: str = ""):
    """orig, recon: (1024, 200) space x time. Each panel uses its own (min, max)."""
    err = orig - recon
    rel = np.linalg.norm(err) / max(np.linalg.norm(orig), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    common = dict(aspect="auto", origin="lower",
                  extent=[0, T_TRAJ, 0, NX], cmap="RdBu_r")

    im0 = axes[0].imshow(orig, vmin=orig.min(), vmax=orig.max(), **common)
    axes[0].set_title(
        f"original  u(t,x){title_suffix}\nshape (1024, 200)  "
        f"range=[{orig.min():.3g}, {orig.max():.3g}]"
    )
    axes[0].set_xlabel("time index (1..200)")
    axes[0].set_ylabel("space index x")

    im1 = axes[1].imshow(recon, vmin=recon.min(), vmax=recon.max(), **common)
    axes[1].set_title(
        f"reconstructed (rank-32 patch SVD)\nRelErr = {rel:.3e}  "
        f"range=[{recon.min():.3g}, {recon.max():.3g}]"
    )
    axes[1].set_xlabel("time index")

    im2 = axes[2].imshow(err, vmin=err.min(), vmax=err.max(), **common)
    axes[2].set_title(
        f"residual = orig - recon\n||err||_2 = {np.linalg.norm(err):.3e}  "
        f"range=[{err.min():.3g}, {err.max():.3g}]"
    )
    axes[2].set_xlabel("time index")

    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    plt.colorbar(im2, ax=axes[2], shrink=0.8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[viz ] traj  RelErr={rel:.3e}  -> {out_path}")


def plot_ic_compare(ic_vec: np.ndarray, recon: np.ndarray, out_path: Path, title_suffix: str = ""):
    """ic_vec: (1024,) original IC. recon: (1024, 20) reconstructed.
    Each panel uses its own |val|.max()."""
    orig_2d = np.broadcast_to(ic_vec[:, None], (NX, PATCH_T))  # (1024, 20)
    err = orig_2d - recon
    rel_2d = np.linalg.norm(err) / max(np.linalg.norm(orig_2d), 1e-12)
    rel_col = np.linalg.norm(ic_vec - recon[:, 0]) / max(np.linalg.norm(ic_vec), 1e-12)

    fig = plt.figure(figsize=(20, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.2])
    common = dict(aspect="auto", origin="lower",
                  extent=[0, PATCH_T, 0, NX], cmap="RdBu_r")

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(orig_2d, vmin=orig_2d.min(), vmax=orig_2d.max(), **common)
    ax0.set_title(
        f"original IC (replicated){title_suffix}\nshape (1024, 20)  "
        f"range=[{orig_2d.min():.3g}, {orig_2d.max():.3g}]"
    )
    ax0.set_xlabel("replicated col idx")
    ax0.set_ylabel("space index x")
    plt.colorbar(im0, ax=ax0, shrink=0.8)

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(recon, vmin=recon.min(), vmax=recon.max(), **common)
    ax1.set_title(
        f"reconstructed (rank-32)\nRelErr (full 2D) = {rel_2d:.3e}  "
        f"range=[{recon.min():.3g}, {recon.max():.3g}]"
    )
    ax1.set_xlabel("col idx")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(err, vmin=err.min(), vmax=err.max(), **common)
    ax2.set_title(
        f"residual\n||err||_2 = {np.linalg.norm(err):.3e}  "
        f"range=[{err.min():.3g}, {err.max():.3g}]"
    )
    ax2.set_xlabel("col idx")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(ic_vec, np.arange(NX), label="original IC", color="black", lw=1.0)
    ax3.plot(recon[:, 0], np.arange(NX), label="recon col 0", color="C3", lw=0.8, ls="--")
    ax3.set_xlabel("u(0, x)")
    ax3.set_ylabel("space index x")
    ax3.set_title(f"1D overlay\nRelErr (col 0) = {rel_col:.3e}")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[viz ] ic    RelErr={rel_2d:.3e}  (col0={rel_col:.3e})  -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_name = "burgers_1d.pt" if args.split == "train" else "burgers_1d_test.pt"
    factor_name = f"burgers_1d_{args.split}.pt"

    print(f"[load] original: {ORIG_DIR}/{orig_name}")
    orig = torch.load(os.path.join(ORIG_DIR, orig_name), map_location="cpu", weights_only=False)
    u = orig["tensor"][args.idx].numpy()                  # (201, 1024)
    nu_val = float(orig["nu"][args.idx])
    print(f"[load] sample idx={args.idx}  u.shape={u.shape}  nu={nu_val:.2e}")

    print(f"[load] factors:  {FACTOR_DIR}/{factor_name}")
    fac = torch.load(os.path.join(FACTOR_DIR, factor_name), map_location="cpu", weights_only=False)
    alpha = fac["alpha"][args.idx].float()                # (320, 32)
    V_hat = fac["V_hat"][args.idx].float()                # (640, 32)
    alpha_ic = fac["alpha_ic"][args.idx].float()          # (32, 32)
    V_hat_ic = fac["V_hat_ic"][args.idx].float()          # (640, 32)

    # ── trajectory: build ground-truth (1024, 200) ─────────────────────────
    traj_orig = u[1:].T                                    # (1024, 200)
    traj_recon = reconstruct_traj_from_factors(alpha, V_hat)  # (1024, 200)

    out_traj = out_dir / f"viz_traj_{args.split}_idx{args.idx:05d}.png"
    plot_traj_compare(traj_orig, traj_recon, out_traj,
                      title_suffix=f"  ({args.split} idx={args.idx}, nu={nu_val:.1e})")

    # ── initial condition: build (1024, 20) by replicating the IC vector ──
    ic_vec = u[0]                                          # (1024,)
    ic_recon = reconstruct_ic_from_factors(alpha_ic, V_hat_ic)  # (1024, 20)

    out_ic = out_dir / f"viz_ic_{args.split}_idx{args.idx:05d}.png"
    plot_ic_compare(ic_vec, ic_recon, out_ic,
                    title_suffix=f"  ({args.split} idx={args.idx}, nu={nu_val:.1e})")

    print("\n[done]")


if __name__ == "__main__":
    main()
