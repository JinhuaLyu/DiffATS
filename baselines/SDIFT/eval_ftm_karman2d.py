import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from FTM_model import Tensor_inr_3D
from karman_dataset import KarmanShardedDataset, get_grid_coords


def project_and_reconstruct(X, U, V, W, U_pinv, V_pinv, W_pinv):
    """X: (B, T, 1, H, W) -> X_hat (same shape) via Tucker projection."""
    G = torch.einsum("im, btmjk->btijk", U_pinv, X)   # (B, T, R1, H, W)
    G = torch.einsum("jn, btinl->btijl", V_pinv, G)   # (B, T, R1, R2, W)
    G = torch.einsum("ko, btijo->btijk", W_pinv, G)   # (B, T, R1, R2, R3)
    Xh = torch.einsum("mi, btijk->btmjk", U, G)       # (B, T, 1, R2, R3)
    Xh = torch.einsum("nj, btmjk->btmnk", V, Xh)      # (B, T, 1, H, R3)
    Xh = torch.einsum("ok, btmnk->btmno", W, Xh)      # (B, T, 1, H, W)
    return Xh, G


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--basis_path", type=str, required=True,
                   help="Path to saved basis_best.pth")
    p.add_argument("--data_root", type=str,
                   default="/projects/p32954/bkx8728/karman_vortex_2d")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--T", type=int, default=201)
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--W", type=int, default=128)
    p.add_argument("--R", type=int, nargs=3, default=(1, 12, 12))
    p.add_argument("--omega", type=float, default=20.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_csv", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    # ---- Load basis ----
    basis = torch.load(args.basis_path, map_location=device, weights_only=False)
    if not isinstance(basis, Tensor_inr_3D):
        # If saved as state_dict, recreate and load
        m = Tensor_inr_3D(tuple(args.R), omega=args.omega).to(device)
        if isinstance(basis, dict):
            m.load_state_dict(basis)
        basis = m
    basis = basis.to(device).eval()
    basis.mode = "training"
    print(f"[basis] loaded from {args.basis_path}", flush=True)

    # ---- Build basis matrices on the full grid ----
    u, v, w = get_grid_coords(H=args.H, W=args.W, device=device)
    with torch.no_grad():
        U, V, W = basis(input_ind_train=(u, v, w))
    U = U.float()
    V = V.float()
    W = W.float()
    print(f"[basis matrices] U={tuple(U.shape)} V={tuple(V.shape)} W={tuple(W.shape)}",
          flush=True)

    U_pinv = torch.linalg.pinv(U)
    V_pinv = torch.linalg.pinv(V)
    W_pinv = torch.linalg.pinv(W)
    print(f"[pinvs] U+={tuple(U_pinv.shape)} V+={tuple(V_pinv.shape)} W+={tuple(W_pinv.shape)}",
          flush=True)

    # ---- Test data ----
    ds = KarmanShardedDataset(args.data_root, split=args.split, T=args.T)
    print(f"[data] {args.split} clips={len(ds)} shards={len(ds.shards)}",
          flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # ---- Eval loop ----
    rel_l1, rel_l2, rmse_norm, plain_rmse = [], [], [], []
    n_seen = 0
    with torch.no_grad():
        for X, idx in loader:
            X = X.to(device, non_blocking=True).float()  # (B, T, 1, H, W)
            B = X.size(0)
            Xh, _ = project_and_reconstruct(X, U, V, W, U_pinv, V_pinv, W_pinv)

            diff = Xh - X
            # Per-trajectory norms
            for i in range(B):
                xi = X[i]
                di = diff[i]
                abs_xi = xi.abs().sum().item()
                abs_di = di.abs().sum().item()
                sq_xi = (xi ** 2).sum().item()
                sq_di = (di ** 2).sum().item()
                if abs_xi == 0 or sq_xi == 0:
                    continue
                rel_l1.append(abs_di / abs_xi)
                rel_l2.append((sq_di ** 0.5) / (sq_xi ** 0.5))
                rmse_norm.append(sq_di / sq_xi)
                # Plain RMSE: sqrt(mean((Xh - X)^2))
                plain_rmse.append((sq_di / xi.numel()) ** 0.5)
            n_seen += B
            print(f"  processed {n_seen}/{len(ds)}", flush=True)

    print()
    print("=" * 56)
    print(f"  trajectories evaluated: {len(rel_l1)}")
    print(f"  Average Relative L1 error : {np.mean(rel_l1):.6e}")
    print(f"  Average Relative L2 error : {np.mean(rel_l2):.6e}")
    print(f"  Average rMSE (||d||^2/||x||^2): {np.mean(rmse_norm):.6e}")
    print(f"  Average RMSE (sqrt mse)   : {np.mean(plain_rmse):.6e}")
    print(f"  std L1 / L2 / rMSE        : {np.std(rel_l1):.4e} / "
          f"{np.std(rel_l2):.4e} / {np.std(rmse_norm):.4e}")
    print("=" * 56)

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["traj_idx", "rel_l1", "rel_l2", "rmse_norm", "rmse"])
            for i, (a, b, c, d) in enumerate(zip(rel_l1, rel_l2, rmse_norm, plain_rmse)):
                wr.writerow([i, a, b, c, d])
        print(f"[csv] wrote per-trajectory metrics to {args.out_csv}")


if __name__ == "__main__":
    main()
