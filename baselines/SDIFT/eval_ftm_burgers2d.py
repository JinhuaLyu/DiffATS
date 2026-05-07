import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim

from FTM_model import Tensor_inr_3D  # noqa: F401  (needed to unpickle basis)
from karman_dataset import KarmanShardedDataset, build_loader, get_grid_coords


def reconstruct(basis_function, tucker_core_slice, ind_input):
    basises = basis_function(input_ind_train=ind_input)  # (U, V, W)
    out = torch.einsum("mi, btijk->btmjk", basises[0], tucker_core_slice)
    out = torch.einsum("nj, btmjk->btmnk", basises[1], out)
    out = torch.einsum("ok, btmnk->btmno", basises[2], out)
    return out


def per_sample_metrics(pred, gt, eps=1e-8):
    """Returns per-sample (rel_l1, rel_l2, rmse) for batched (B, T, 1, H, W)."""
    B = pred.size(0)
    diff = pred - gt
    diff_flat = diff.reshape(B, -1)
    gt_flat = gt.reshape(B, -1)
    rel_l1 = diff_flat.abs().sum(-1) / (gt_flat.abs().sum(-1) + eps)
    rel_l2 = diff_flat.pow(2).sum(-1).sqrt() / (gt_flat.pow(2).sum(-1).sqrt() + eps)
    rmse = diff_flat.pow(2).mean(-1).sqrt()
    return rel_l1, rel_l2, rmse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--basis_path", type=str, required=True,
                   help="Path to basis_best.pth saved by train_FTM_karman.py")
    p.add_argument("--data_root", type=str,
                   default="${DATA_ROOT}/burgers_2d")
    p.add_argument("--out_dir", type=str,
                   default="${DATA_ROOT}/bkx8728/burgers_sdift_runs/eval_ftm_test")
    p.add_argument("--T", type=int, default=201)
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--W", type=int, default=128)
    p.add_argument("--R", type=int, nargs=3, default=(1, 9, 9))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000,
                   help="Adam steps per batch to fit fresh Tucker cores")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_clips", type=int, default=None,
                   help="Cap test samples (default: full 1000)")
    p.add_argument("--seed", type=int, default=231)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] device={device}")
    print(f"[eval] basis_path={args.basis_path}")
    print(f"[eval] out_dir={out_dir}")

    # Load basis (saved as full module via torch.save(basis_function, ...))
    basis_function = torch.load(args.basis_path, map_location=device,
                                weights_only=False)
    basis_function.eval()
    basis_function.mode = "training"  # forward expects (U, V, W) ind tuple
    for p in basis_function.parameters():
        p.requires_grad_(False)
    n_basis = sum(p.numel() for p in basis_function.parameters())
    print(f"[eval] basis params={n_basis:,}  R={tuple(args.R)}  omega=20")

    # Coords (single-channel: u_ind_uni length 1)
    u, v, w = get_grid_coords(H=args.H, W=args.W, device=device)
    ind_input = (u, v, w)

    # Test dataset (split='test' uses ${DATA_ROOT}/burgers_2d_test)
    test_ds = KarmanShardedDataset(args.data_root, split='test',
                                   T=args.T, max_clips=args.max_clips)
    test_loader = build_loader(test_ds, batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=False, drop_last=False)
    print(f"[eval] N_test={len(test_ds)}  batch_size={args.batch_size}  "
          f"steps_per_batch={args.steps}  lr={args.lr}")

    all_l1, all_l2, all_rmse = [], [], []
    t0 = time.time()

    for batch_idx, (clip_batch, idx_batch) in enumerate(test_loader):
        clip_batch = clip_batch.to(device, non_blocking=True)
        B = clip_batch.shape[0]

        # Fresh learnable core per batch, identical init to training
        core = (torch.ones(B, args.T, args.R[0], args.R[1], args.R[2],
                           device=device) / 2.0).requires_grad_(True)
        opt = optim.AdamW([core], lr=args.lr)

        last_loss = float('nan')
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            pred = reconstruct(basis_function, core, ind_input)
            loss = (pred - clip_batch).pow(2).mean()
            loss.backward()
            opt.step()
            last_loss = loss.item()

        with torch.no_grad():
            pred = reconstruct(basis_function, core, ind_input)
            rel_l1, rel_l2, rmse_val = per_sample_metrics(pred, clip_batch)
            all_l1.append(rel_l1.detach().cpu().numpy())
            all_l2.append(rel_l2.detach().cpu().numpy())
            all_rmse.append(rmse_val.detach().cpu().numpy())

        n_done = sum(len(x) for x in all_l1)
        elapsed = time.time() - t0
        eta = elapsed / max(n_done, 1) * (len(test_ds) - n_done)
        print(f"[batch {batch_idx+1:3d}] done={n_done:4d}/{len(test_ds)}  "
              f"final_mse={last_loss:.6f}  "
              f"rL1={rel_l1.mean():.5f}  rL2={rel_l2.mean():.5f}  "
              f"rmse={rmse_val.mean():.5e}  "
              f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    all_l1 = np.concatenate(all_l1)
    all_l2 = np.concatenate(all_l2)
    all_rmse = np.concatenate(all_rmse)
    n = len(all_l1)

    def fmt(arr):
        m = float(arr.mean())
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        return m, sem

    l1_m, l1_s = fmt(all_l1)
    l2_m, l2_s = fmt(all_l2)
    rmse_m, rmse_s = fmt(all_rmse)

    print()
    print("=" * 70)
    print(f"FTM Test-Set Reconstruction (N={n})")
    print("=" * 70)
    print(f"Average Relative L1 : {l1_m:.4f} ± {l1_s:.4f}")
    print(f"Average Relative L2 : {l2_m:.4f} ± {l2_s:.4f}")
    print(f"Average rMSE        : {rmse_m:.4e} ± {rmse_s:.2e}")
    print("=" * 70)

    np.savez(out_dir / "ftm_test_eval.npz",
             rel_l1=all_l1, rel_l2=all_l2, rmse=all_rmse,
             l1_mean=l1_m, l1_sem=l1_s,
             l2_mean=l2_m, l2_sem=l2_s,
             rmse_mean=rmse_m, rmse_sem=rmse_s,
             n=n, args=vars(args))
    print(f"[saved] {out_dir}/ftm_test_eval.npz")


if __name__ == "__main__":
    main()
