import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as scio
import torch
from torch import optim
from tqdm import tqdm

from FTM_model import Tensor_inr_3D
from karman_dataset import KarmanShardedDataset, build_loader, get_grid_coords
from utils import total_variation_loss


def loss_rmse(pred, gt):
    return torch.sqrt(torch.mean((pred - gt) ** 2))


def loss_mae(pred, gt):
    return torch.mean(torch.abs(pred - gt))


def reconstruct(basis_function, tucker_core_slice):
    """Reconstruct (B, T, 1, H, W) from cores (B, T, R1, R2, R3) and basis."""
    basises = basis_function(input_ind_train=ind_input_global)
    out = torch.einsum("mi, btijk->btmjk", basises[0], tucker_core_slice)
    out = torch.einsum("nj, btmjk->btmnk", basises[1], out)
    out = torch.einsum("ok, btmnk->btmno", basises[2], out)
    return out


def train_one_epoch(basis_function, tucker_core, train_loader, optimizer,
                    args, device, epoch, log_every=20):
    basis_function.train()
    basis_function.mode = "training"
    losses = []
    step_times = []
    t0 = time.time()
    pbar = tqdm(train_loader, desc=f"FTM epoch {epoch}", leave=False)
    for it, (clip_batch, batch_idx) in enumerate(pbar):
        step_t0 = time.time()
        clip_batch = clip_batch.to(device, non_blocking=True)
        batch_idx = batch_idx.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        core_slice = tucker_core[batch_idx]
        pred = reconstruct(basis_function, core_slice)
        loss = loss_rmse(pred, clip_batch) + total_variation_loss(
            tucker_core[batch_idx], weight=args.tv_weight)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        step_times.append(time.time() - step_t0)
        if it % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             step_ms=f"{step_times[-1]*1000:.0f}")

    epoch_t = time.time() - t0
    mean_step = np.mean(step_times) * 1000 if step_times else 0
    return float(np.mean(losses)), epoch_t, mean_step


@torch.no_grad()
def evaluate(basis_function, tucker_core, eval_loader, device):
    basis_function.eval()
    rmse_list, mae_list = [], []
    for clip_batch, batch_idx in eval_loader:
        clip_batch = clip_batch.to(device, non_blocking=True)
        batch_idx = batch_idx.to(device, non_blocking=True)
        core_slice = tucker_core[batch_idx]
        pred = reconstruct(basis_function, core_slice)
        rmse_list.append(loss_rmse(pred, clip_batch).item())
        mae_list.append(loss_mae(pred, clip_batch).item())
    return float(np.mean(rmse_list)), float(np.mean(mae_list))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--data_name", type=str, default="karman2d")
    p.add_argument("--out_dir", type=str, default="./runs")
    p.add_argument("--T", type=int, default=201)
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--W", type=int, default=128)
    p.add_argument("--R", type=int, nargs=3, default=(1, 12, 12))
    p.add_argument("--omega", type=float, default=20.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max_clips", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--tv_weight", type=float, default=1e-7)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=231)
    return p.parse_args()


def main():
    global ind_input_global

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"ftm_{args.data_name}_{'x'.join(map(str, args.R))}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")
    print(f"[run_dir] {run_dir}")

    train_ds = KarmanShardedDataset(
        args.data_root, split='train', T=args.T, max_clips=args.max_clips)
    train_loader = build_loader(
        train_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True, drop_last=True)
    print(f"[data] train clips={len(train_ds)}  shards={len(train_ds.shards)}  "
          f"batches/epoch={len(train_loader)}")

    eval_ds = KarmanShardedDataset(
        args.data_root, split='train', T=args.T, max_clips=128)
    eval_loader = build_loader(
        eval_ds, batch_size=args.batch_size,
        num_workers=2, shuffle=False, drop_last=False)

    u, v, w = get_grid_coords(H=args.H, W=args.W, device=device)
    ind_input_global = (u, v, w)

    R = tuple(args.R)
    basis_function = Tensor_inr_3D(R, omega=args.omega).to(device)
    n_params = sum(p.numel() for p in basis_function.parameters())
    print(f"[model] FTM params: {n_params:,}  (R={R}, omega={args.omega})")

    N = len(train_ds)
    tucker_core = (torch.ones(N, args.T, R[0], R[1], R[2], device=device) / 2.0)
    tucker_core.requires_grad_(True)
    print(f"[model] tucker_core shape: {tuple(tucker_core.shape)}  "
          f"size: {tucker_core.numel() * 4 / 1e9:.2f} GB")

    params = list(basis_function.parameters()) + [tucker_core]
    optimizer = optim.AdamW(params, lr=args.lr)

    history = {"epoch": [], "train_rmse": [], "eval_rmse": [], "eval_mae": [],
               "epoch_seconds": [], "mean_step_ms": []}
    best_rmse = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loader.batch_sampler.set_epoch(epoch)
        train_rmse, ep_t, mean_step = train_one_epoch(
            basis_function, tucker_core, train_loader, optimizer,
            args, device, epoch)
        msg = (f"[epoch {epoch:3d}] train_rmse={train_rmse:.5f} "
               f"epoch_time={ep_t:.1f}s  mean_step={mean_step:.0f}ms")
        history["epoch"].append(epoch)
        history["train_rmse"].append(train_rmse)
        history["epoch_seconds"].append(ep_t)
        history["mean_step_ms"].append(mean_step)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            rmse, mae = evaluate(basis_function, tucker_core, eval_loader, device)
            history["eval_rmse"].append(rmse)
            history["eval_mae"].append(mae)
            msg += f"  eval_rmse={rmse:.5f}  eval_mae={mae:.5f}"
            if rmse < best_rmse:
                best_rmse = rmse
                scio.savemat(str(run_dir / "core_best.mat"),
                             {"core": tucker_core.detach().cpu().numpy()})
                torch.save(basis_function, run_dir / "basis_best.pth")
                print(f"  -> saved best (rmse={rmse:.5f})")

        print(msg, flush=True)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({
                "epoch": epoch,
                "basis_state": basis_function.state_dict(),
                "tucker_core": tucker_core.detach().cpu(),
                "args": vars(args),
            }, run_dir / f"ckpt_epoch_{epoch:03d}.pt")
            scio.savemat(str(run_dir / f"core_epoch_{epoch:03d}.mat"),
                         {"core": tucker_core.detach().cpu().numpy()})

    np.save(run_dir / "history.npy", history, allow_pickle=True)
    print(f"[done] best_rmse={best_rmse:.5f}  run_dir={run_dir}")


if __name__ == "__main__":
    main()
