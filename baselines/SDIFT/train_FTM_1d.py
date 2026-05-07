import argparse
import os
import time
from datetime import datetime

import numpy as np
import scipy.io as scio
import torch
from torch import optim
from tqdm import tqdm

from FTM_model_1d import Tensor_inr_1D
from utils_1d import BurgersPTDataset, Reaction1DPTDataset, total_variation_loss_1d


def loss_rmse(pred, gt):
    diff = pred - gt
    return torch.sqrt(torch.mean(diff * diff))


def loss_mae(pred, gt):
    return torch.mean(torch.abs(pred - gt))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_name", type=str, default="burgers_1d")
    p.add_argument(
        "--dataset_type",
        type=str,
        choices=["burgers", "reaction"],
        default="burgers",
        help="burgers: one scalar (nu). reaction: two scalars (nu, rho).",
    )
    p.add_argument(
        "--data_path",
        type=str,
        default="/scratch/bkx8728/burgers_1d/burgers_1d.pt",
        help="Path to the .pt file containing 'tensor', 'nu' [, 'rho'], 'x_coord', 't_coord'.",
    )
    p.add_argument("--R1", type=int, default=64, help="Spatial rank (must be a multiple of UNet downsampling factor).")
    p.add_argument("--omega", type=float, default=20.0)
    p.add_argument("--mid_channel", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_iter", type=int, default=2000, help="Number of epochs over the dataset.")
    p.add_argument("--tv_weight", type=float, default=1e-7)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--start_save_iter", type=int, default=50)
    p.add_argument("--ckp_dir", type=str, default="./ckp")
    p.add_argument("--core_dir", type=str, default="./data")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max_samples", type=int, default=-1, help="Truncate dataset for smoke tests; -1 = use all.")
    p.add_argument("--core_init_scale", type=float, default=0.5)
    return p.parse_args()


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print("device:", device)
    os.makedirs(cfg.ckp_dir, exist_ok=True)
    os.makedirs(cfg.core_dir, exist_ok=True)

    if cfg.dataset_type == "burgers":
        base = BurgersPTDataset(cfg.data_path)
    else:
        base = Reaction1DPTDataset(cfg.data_path)
    if cfg.max_samples > 0 and cfg.max_samples < len(base):
        ds = torch.utils.data.Subset(base, list(range(cfg.max_samples)))
        N = cfg.max_samples
        nu_full = base.nu[:N].clone()
        rho_full = base.rho[:N].clone() if cfg.dataset_type == "reaction" else None
    else:
        ds = base
        N = len(base)
        nu_full = base.nu.clone()
        rho_full = base.rho.clone() if cfg.dataset_type == "reaction" else None
    T, X = base.T, base.X
    x_coord = base.x_coord.clone().to(device)

    print(f"data: N={N}, T={T}, X={X} ({cfg.dataset_type})")

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    R1 = cfg.R1
    basis = Tensor_inr_1D(R1=R1, omega=cfg.omega, mid_channel=cfg.mid_channel).to(device)

    # Per-sample, per-timestep core; lives on GPU as one big parameter tensor.
    tucker_core = (
        torch.ones(N, T, R1, device=device) * cfg.core_init_scale
    )
    tucker_core.requires_grad = True

    params = list(basis.parameters()) + [tucker_core]
    opt = optim.AdamW(params, lr=cfg.lr)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    rmse_min = float("inf")

    for epoch in range(cfg.max_iter):
        basis.train()
        basis.mode = "training"
        t0 = time.time()
        losses = []
        for batch in loader:
            # burgers: (u_tx, nu, idx); reaction: (u_tx, nu, rho, idx).
            u_tx = batch[0].to(device, non_blocking=True)
            idx = batch[-1].to(device)
            opt.zero_grad(set_to_none=True)

            U = basis(x_coord)  # (X, R1)
            recon = torch.einsum("xr, btr -> btx", U, tucker_core[idx])
            loss = loss_rmse(recon, u_tx) + total_variation_loss_1d(
                tucker_core[idx], cfg.tv_weight
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())
        loss_mean = float(np.mean(losses))
        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == cfg.max_iter - 1:
            print(f"epoch {epoch:5d} | loss={loss_mean:.5f} | {elapsed:.1f}s")

        do_save = (
            epoch >= cfg.start_save_iter
            and (loss_mean < rmse_min or epoch % cfg.save_every == 0)
        )
        if do_save:
            rmse_min = min(rmse_min, loss_mean)
            basis_path = f"{cfg.ckp_dir}/basis_{cfg.data_name}_R{R1}_{ts}.pth"
            core_path = f"{cfg.core_dir}/core_{cfg.data_name}_R{R1}_{ts}.mat"
            torch.save(basis, basis_path)
            mat = {
                "core": tucker_core.detach().cpu().numpy(),  # (N, T, R1)
                "nu": nu_full.numpy(),
                "x_coord": base.x_coord.numpy(),
                "t_coord": base.t_coord.numpy(),
                "loss_mean": float(loss_mean),
                "epoch": int(epoch),
            }
            if rho_full is not None:
                mat["rho"] = rho_full.numpy()
            scio.savemat(core_path, mat)
            print(f"  saved: {core_path}")


if __name__ == "__main__":
    main()
