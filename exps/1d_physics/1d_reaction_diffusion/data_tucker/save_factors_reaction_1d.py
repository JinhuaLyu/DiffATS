"""save_factors_reaction_1d.py — patch-SVD + Procrustes alignment for 1D RD.

Pipeline (per sample, trajectory u(t,x) of shape (201, 1024)):

  1. Trajectory frames t=1..200 -> reshape to (NX=1024, T=200) (space x time).
     Patchify with patch (PATCH_X=32, PATCH_T=20):
        N_patches = (1024/32)*(200/20) = 320
        patch_dim = 32*20 = 640
     Row-major order: outer = spatial block (32), inner = time block (10).
     Result: A (320, 640).
  2. Full SVD A = U S V^T, truncate to RANK=32.
  3. One reference sample (chosen by REF_SEED=42 from training set) provides
     V_anchor (640, 32). All other samples Procrustes-align V to V_anchor:
        Q = procrustes(V_r, V_anchor)
        alpha = (U_r * S_r) @ Q       (320, 32)
        V_hat = V_r @ Q                (640, 32)
     so that alpha @ V_hat^T == U_r diag(S_r) V_r^T (rank-r truncation).
  4. Initial frame u(0, .) of shape (1024,):
        chunks = u(0).reshape(32, 32)            # 32 chunks of length 32
        replicate each chunk 20x along time      # -> (32, 32, 20)
        flatten patches (row-major: spatial,time) -> A_ic (32, 640)
     Full SVD A_ic = U_ic S_ic Vh_ic, naturally rank 32.
     Procrustes-align V_ic to the SAME V_anchor:
        Q_ic = procrustes(V_ic, V_anchor)
        alpha_ic = (U_ic * S_ic) @ Q_ic   (32, 32)
        V_hat_ic = V_ic @ Q_ic            (640, 32)

Both alpha_ic and V_hat_ic are stored as conditioning (mirrors the 2D Burgers
Tucker pipeline where U_ic and Vh_ic both feed into the diffusion model).

Output: a single .pt per split, plus ref_anchor.pt at <out_dir>.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


PATCH_X = 32
PATCH_T = 20
NX = 1024
T_TRAJ = 200          # frames t=1..200 (skip t=0)
N_BLOCK_X = NX // PATCH_X       # 32
N_BLOCK_T = T_TRAJ // PATCH_T   # 10
N_PATCHES = N_BLOCK_X * N_BLOCK_T  # 320
PATCH_DIM = PATCH_X * PATCH_T      # 640
RANK = 32
REF_SEED = 42

IC_CHUNK = 32                     # 1024 / 32 = 32 chunks of length 32
N_PATCHES_IC = NX // IC_CHUNK     # 32


# ---------------------------------------------------------------------------
# Patchification
# ---------------------------------------------------------------------------

def patchify_traj(traj: torch.Tensor) -> torch.Tensor:
    """traj: (B, T_TRAJ, NX). Returns (B, N_PATCHES, PATCH_DIM).

    Row-major patch order: spatial-block (slow), time-block (fast).
    Within each patch, axis order is (PATCH_X, PATCH_T) flattened row-major
    (spatial slow, time fast).
    """
    assert traj.shape[-2] == T_TRAJ and traj.shape[-1] == NX, \
        f"unexpected traj shape {tuple(traj.shape)}"
    # (B, T, NX) -> (B, NX, T)
    ust = traj.transpose(-1, -2).contiguous()
    # unfold: spatial dim then time dim
    A = (
        ust.unfold(-2, PATCH_X, PATCH_X)   # (B, N_BLOCK_X, T, PATCH_X)
           .unfold(-2, PATCH_T, PATCH_T)   # (B, N_BLOCK_X, N_BLOCK_T, PATCH_X, PATCH_T)
           .contiguous()
           .reshape(traj.shape[0], N_PATCHES, PATCH_DIM)
    )
    return A


def patchify_ic(ic: torch.Tensor) -> torch.Tensor:
    """ic: (B, NX). Returns (B, N_PATCHES_IC, PATCH_DIM).

    Each chunk of length IC_CHUNK is replicated PATCH_T times along the time
    axis to form a (PATCH_X, PATCH_T) patch, then flattened row-major
    (spatial slow, time fast) to align with `patchify_traj`.
    """
    assert ic.shape[-1] == NX, f"unexpected ic shape {tuple(ic.shape)}"
    chunks = ic.reshape(ic.shape[0], N_PATCHES_IC, IC_CHUNK)        # (B, 32, 32)
    patches = chunks.unsqueeze(-1).expand(-1, -1, IC_CHUNK, PATCH_T)  # (B, 32, 32, 20)
    A_ic = patches.reshape(ic.shape[0], N_PATCHES_IC, PATCH_DIM)    # (B, 32, 640)
    return A_ic


# ---------------------------------------------------------------------------
# SVD + Procrustes
# ---------------------------------------------------------------------------

def truncated_svd(A: torch.Tensor, rank: int):
    """Full SVD then truncate. A: (..., m, n).
    Returns U_r (..., m, r), S_r (..., r), V_r (..., n, r).
    """
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return U[..., :, :rank], S[..., :rank], Vh[..., :rank, :].transpose(-1, -2).contiguous()


def procrustes_align(V_r: torch.Tensor, V_anchor: torch.Tensor) -> torch.Tensor:
    """Find orthogonal Q (..., r, r) minimising ||V_r @ Q - V_anchor||_F.
    V_r: (..., n, r); V_anchor: (n, r)  (broadcast over leading dims).
    """
    M = V_r.transpose(-1, -2) @ V_anchor                # (..., r, r)
    Up, _, Vhp = torch.linalg.svd(M, full_matrices=False)
    return Up @ Vhp                                      # (..., r, r)


def factorize_batch(A: torch.Tensor, V_anchor: torch.Tensor, rank: int):
    """A: (B, m, n). Returns alpha (B, m, r), V_hat (B, n, r)."""
    U_r, S_r, V_r = truncated_svd(A, rank)               # (B,m,r) (B,r) (B,n,r)
    Q = procrustes_align(V_r, V_anchor)                  # (B, r, r)
    V_hat = V_r @ Q                                      # (B, n, r)
    alpha = (U_r * S_r.unsqueeze(-2)) @ Q                # (B, m, r)
    return alpha, V_hat


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def process_split(
    in_path: Path,
    V_anchor: torch.Tensor,
    rank: int,
    batch_size: int,
    device: torch.device,
):
    print(f"[load] {in_path}", flush=True)
    payload = torch.load(in_path, map_location="cpu", weights_only=False)
    tensor = payload["tensor"]            # (N, 201, 1024) float32
    nu = payload["nu"]                    # (N,)
    rho = payload.get("rho")              # (N,) for RD; absent for Burgers
    N = tensor.shape[0]
    print(f"[load] N={N}  tensor.shape={tuple(tensor.shape)}", flush=True)

    alpha_all = torch.empty((N, N_PATCHES, rank), dtype=torch.float32)
    Vhat_all = torch.empty((N, PATCH_DIM, rank), dtype=torch.float32)
    alpha_ic_all = torch.empty((N, N_PATCHES_IC, rank), dtype=torch.float32)
    Vhat_ic_all = torch.empty((N, PATCH_DIM, rank), dtype=torch.float32)

    V_anchor_dev = V_anchor.to(device)

    t0 = time.time()
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        chunk = tensor[s:e].to(device, non_blocking=True)   # (b, 201, 1024)
        traj = chunk[:, 1:, :]                              # (b, 200, 1024)
        ic = chunk[:, 0, :]                                 # (b, 1024)

        A = patchify_traj(traj)                              # (b, 320, 640)
        A_ic = patchify_ic(ic)                               # (b, 32, 640)

        alpha, V_hat = factorize_batch(A, V_anchor_dev, rank)
        alpha_ic, V_hat_ic = factorize_batch(A_ic, V_anchor_dev, rank)

        alpha_all[s:e] = alpha.detach().cpu()
        Vhat_all[s:e] = V_hat.detach().cpu()
        alpha_ic_all[s:e] = alpha_ic.detach().cpu()
        Vhat_ic_all[s:e] = V_hat_ic.detach().cpu()

        elapsed = time.time() - t0
        eta = elapsed / (e) * (N - e) if e > 0 else 0.0
        print(
            f"[run ] [{s:>5d},{e:>5d})/{N}  "
            f"elapsed={elapsed:.1f}s  eta={eta:.1f}s",
            flush=True,
        )

    print(f"[run ] total {time.time() - t0:.1f}s", flush=True)
    return {
        "alpha":     alpha_all,
        "V_hat":     Vhat_all,
        "alpha_ic":  alpha_ic_all,
        "V_hat_ic":  Vhat_ic_all,
        "nu":        nu.float() if isinstance(nu, torch.Tensor) else torch.tensor(nu, dtype=torch.float32),
        "rho":       (rho.float() if isinstance(rho, torch.Tensor)
                      else (torch.tensor(rho, dtype=torch.float32) if rho is not None else None)),
        "x_coord":   payload.get("x_coord"),
        "t_coord":   payload.get("t_coord"),
        "meta_in":   payload.get("meta", {}),
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_check(in_path: Path, out_payload: dict, idx: int = 0):
    print(f"\n[chk ] sanity check on idx={idx} from {in_path.name}", flush=True)
    raw = torch.load(in_path, map_location="cpu", weights_only=False)
    u = raw["tensor"][idx].to(torch.float64)            # (201, 1024)
    traj = u[1:]                                         # (200, 1024)
    ic = u[0]                                            # (1024,)

    A_true = patchify_traj(traj.unsqueeze(0).float()).squeeze(0).to(torch.float64)
    alpha = out_payload["alpha"][idx].to(torch.float64)            # (320, 32)
    V_hat = out_payload["V_hat"][idx].to(torch.float64)            # (640, 32)
    A_recon = alpha @ V_hat.T                            # (320, 640)
    rel_traj = (A_true - A_recon).norm() / max(A_true.norm().item(), 1e-12)
    print(f"[chk ] traj rank-{RANK} reconstruction RelErr = {rel_traj.item():.4e}")

    A_ic_true = patchify_ic(ic.unsqueeze(0).float()).squeeze(0).to(torch.float64)
    alpha_ic = out_payload["alpha_ic"][idx].to(torch.float64)
    V_hat_ic = out_payload["V_hat_ic"][idx].to(torch.float64)
    A_ic_recon = alpha_ic @ V_hat_ic.T
    rel_ic = (A_ic_true - A_ic_recon).norm() / max(A_ic_true.norm().item(), 1e-12)
    print(f"[chk ] ic   rank-{RANK} reconstruction RelErr = {rel_ic.item():.4e}  (expect ~0 since rank=min(m,n))")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--test_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--rank", type=int, default=RANK)
    p.add_argument("--ref_seed", type=int, default=REF_SEED)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu; auto-detect if unset")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device = {device}", flush=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    # ── Build / load anchor from training set ─────────────────────────────
    anchor_path = out_dir / "ref_anchor.pt"
    if anchor_path.exists():
        anchor = torch.load(anchor_path, map_location="cpu", weights_only=False)
        V_anchor = anchor["V_anchor"].float()
        ref_idx = int(anchor["ref_idx"])
        assert int(anchor["rank"]) == args.rank, "anchor rank mismatch"
        print(f"[anch] loaded existing anchor (ref_idx={ref_idx})  V_anchor={tuple(V_anchor.shape)}", flush=True)
    else:
        print(f"[anch] computing new anchor from {args.train_path}", flush=True)
        train_payload = torch.load(args.train_path, map_location="cpu", weights_only=False)
        N_train = train_payload["tensor"].shape[0]
        rng = random.Random(args.ref_seed)
        ref_idx = rng.randrange(N_train)
        ref_traj = train_payload["tensor"][ref_idx, 1:, :].to(device)   # (200, 1024)
        A_ref = patchify_traj(ref_traj.unsqueeze(0)).squeeze(0)           # (320, 640)
        _, _, V_ref = truncated_svd(A_ref, args.rank)                     # (640, rank)
        V_anchor = V_ref.detach().cpu().float()
        torch.save(
            {
                "V_anchor": V_anchor,
                "ref_idx": int(ref_idx),
                "ref_seed": int(args.ref_seed),
                "rank": int(args.rank),
                "patch_x": int(PATCH_X),
                "patch_t": int(PATCH_T),
                "n_patches": int(N_PATCHES),
                "patch_dim": int(PATCH_DIM),
                "n_train_total": int(N_train),
                "source_train_path": str(args.train_path),
            },
            anchor_path,
        )
        print(f"[anch] saved -> {anchor_path}  ref_idx={ref_idx}/{N_train}", flush=True)
        del train_payload, ref_traj, A_ref

    # ── Process train and test ────────────────────────────────────────────
    for split, in_path in [("train", args.train_path), ("test", args.test_path)]:
        in_path = Path(in_path)
        out_path = out_dir / f"reaction_1d_{split}.pt"
        print(f"\n[==={split}===] in={in_path}  out={out_path}", flush=True)

        result = process_split(
            in_path=in_path,
            V_anchor=V_anchor,
            rank=args.rank,
            batch_size=args.batch_size,
            device=device,
        )
        sanity_check(in_path, result, idx=0)

        result["meta"] = {
            "rank": int(args.rank),
            "patch_x": int(PATCH_X),
            "patch_t": int(PATCH_T),
            "n_patches": int(N_PATCHES),
            "patch_dim": int(PATCH_DIM),
            "n_patches_ic": int(N_PATCHES_IC),
            "ref_seed": int(args.ref_seed),
            "anchor_path": str(anchor_path),
            "split": split,
            "source_path": str(in_path),
            "no_mean_centering": True,
            "patch_order": "row-major (spatial slow, time fast)",
        }
        torch.save(result, out_path)
        size_gb = out_path.stat().st_size / 1024 ** 3
        print(f"[save] {out_path}  ({size_gb:.2f} GB)", flush=True)

    print("\n[done]")


if __name__ == "__main__":
    main()
