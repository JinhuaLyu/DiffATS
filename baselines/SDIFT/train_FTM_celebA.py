import os
import argparse
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from FTM_model import Tensor_inr_3D
from utils import CelebAHQDataset


def loss_fn(pred, gt):
    diff = pred - gt
    return torch.sqrt(torch.mean(diff ** 2))


def loss_fn2(pred, gt):
    return torch.mean(torch.abs(pred - gt))


def core_tv_loss(core_batch, weight):
    # Spatial total variation on the (R1, R2, R3) core to keep it smooth.
    diff_r1 = core_batch[:, :, 1:, :, :] - core_batch[:, :, :-1, :, :]
    diff_r2 = core_batch[:, :, :, 1:, :] - core_batch[:, :, :, :-1, :]
    return weight * (diff_r1.pow(2).mean() + diff_r2.pow(2).mean())


def reconstruct(basis_function, core_batch, ind_input):
    basises = basis_function(input_ind_train=ind_input)  # (U_basis, V_basis, W_basis)
    out = torch.einsum("mi, btijk->btmjk", basises[0], core_batch)
    out = torch.einsum("nj, btmjk->btmnk", basises[1], out)
    out = torch.einsum("ok, btmnk->btmno", basises[2], out)
    return out


class CPUCoreAdam:
    """Adam over per-image cores stored on CPU.

    The cores tensor and its (exp_avg, exp_avg_sq) state live as plain CPU
    tensors. Each step we ship the batch slice to the GPU, accumulate Adam
    moments + apply the update there, and write the slice back to CPU. The
    update formula is identical to torch.optim.Adam.
    """

    def __init__(self, n_images, R, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, init_value=0.5):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        # Shape: (N, T=1, R1, R2, R3) — keep T explicit so reconstruct() works as-is.
        self.cores = torch.full((n_images, 1, R[0], R[1], R[2]), float(init_value), dtype=torch.float32)
        self.exp_avg = torch.zeros_like(self.cores)
        self.exp_avg_sq = torch.zeros_like(self.cores)

    def fetch(self, batch_ind, device):
        """Pull the batch slice to GPU and return a leaf tensor with grad."""
        slice_cpu = self.cores[batch_ind]
        slice_gpu = slice_cpu.to(device, non_blocking=True).detach()
        slice_gpu.requires_grad_(True)
        return slice_gpu

    def step(self, batch_ind, core_slice_gpu):
        """Apply one Adam step using core_slice_gpu.grad and write state back to CPU."""
        with torch.no_grad():
            self.step_count += 1
            grad = core_slice_gpu.grad
            device = grad.device

            m = self.exp_avg[batch_ind].to(device, non_blocking=True)
            v = self.exp_avg_sq[batch_ind].to(device, non_blocking=True)

            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            bc1 = 1 - self.beta1 ** self.step_count
            bc2 = 1 - self.beta2 ** self.step_count
            update = (m / bc1) / ((v / bc2).sqrt() + self.eps)

            new_core = core_slice_gpu.detach() - self.lr * update

            self.cores[batch_ind] = new_core.cpu()
            self.exp_avg[batch_ind] = m.cpu()
            self.exp_avg_sq[batch_ind] = v.cpu()


def train_one_epoch(basis_function, core_optim, loader, optimizer, device, ind_input, tv_weight, log_every=50):
    basis_function.train()
    basis_function.mode = "training"
    losses = []
    for step, (data, batch_ind) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        # Pull batch's cores from CPU
        core_batch = core_optim.fetch(batch_ind, device)

        optimizer.zero_grad(set_to_none=True)
        out = reconstruct(basis_function, core_batch, ind_input)
        loss = loss_fn(out, data) + core_tv_loss(core_batch, weight=tv_weight)
        loss.backward()
        optimizer.step()
        # Apply Adam to the per-image cores using the gradient just computed.
        core_optim.step(batch_ind, core_batch)

        losses.append(loss.item())
        if step % log_every == 0:
            print(f"  step {step}/{len(loader)}  loss={loss.item():.6f}")
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(basis_function, core_optim, loader, device, ind_input, max_batches=4):
    basis_function.eval()
    rmse_acc, mae_acc, n = 0.0, 0.0, 0
    for i, (data, batch_ind) in enumerate(loader):
        if i >= max_batches:
            break
        data = data.to(device, non_blocking=True)
        core_batch = core_optim.cores[batch_ind].to(device, non_blocking=True)
        out = reconstruct(basis_function, core_batch, ind_input)
        rmse_acc += loss_fn(out, data).item() * data.shape[0]
        mae_acc += loss_fn2(out, data).item() * data.shape[0]
        n += data.shape[0]
    return rmse_acc / n, mae_acc / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="celebahq")
    parser.add_argument("--image_root", type=str, default="../CelebA-HQ/celeba_hq_images/all")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--R", type=int, nargs=3, default=(256, 256, 3),
                        help="Tucker core ranks (R1, R2, R3). Default gives 196,608 floats per image.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tv_weight", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=2, help="Save core+basis every N epochs.")
    parser.add_argument("--device", type=str, default="cuda:0")
    config = parser.parse_args()

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---------- dataset ----------
    dataset = CelebAHQDataset(config.image_root, image_size=config.image_size)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(config.num_workers > 0),
    )
    n_images = len(dataset)
    print(f"dataset: {n_images} images @ {config.image_size}x{config.image_size}")

    # ---------- spatial index grids (continuous coords in [0, 1]) ----------
    H = W = config.image_size
    C = 3
    u_ind = torch.linspace(0, 1, H, device=device)
    v_ind = torch.linspace(0, 1, W, device=device)
    w_ind = torch.linspace(0, 1, C, device=device)
    ind_input = (u_ind, v_ind, w_ind)

    # ---------- model + per-image cores ----------
    R = tuple(config.R)
    print("R:", R, "core floats/image:", R[0] * R[1] * R[2])
    basis_function = Tensor_inr_3D(R, omega=20).to(device)

    # Per-image learnable cores live on CPU (~22.6 GB) with their Adam state
    # (~45 GB more) to keep H100 80 GB free for the basis MLPs + activations.
    core_optim = CPUCoreAdam(n_images, R, lr=config.learning_rate)

    n_basis_params = sum(p.numel() for p in basis_function.parameters() if p.requires_grad)
    print(f"FTM basis params: {n_basis_params:,}")
    print(f"FTM core params:  {core_optim.cores.numel():,} (CPU-resident per-image latents)")

    # Adam (on GPU) only sees the basis MLPs.
    optimizer = optim.Adam(basis_function.parameters(), lr=config.learning_rate)

    # ---------- output dirs ----------
    os.makedirs("./ckp", exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    stamp = datetime.now().strftime("%Y_%m_%d_%H")
    tag = f"{config.data_name}_{R[0]}x{R[1]}x{R[2]}_{stamp}"

    # ---------- train ----------
    rmse0, mae0 = evaluate(basis_function, core_optim, loader, device, ind_input)
    print(f"init  RMSE={rmse0:.5f}  MAE={mae0:.5f}")

    best_rmse = float("inf")
    for epoch in tqdm(range(config.num_epochs), desc="epochs"):
        train_loss = train_one_epoch(
            basis_function, core_optim, loader, optimizer, device, ind_input, config.tv_weight,
        )
        print(f"[epoch {epoch}] train_loss={train_loss:.6f}")

        if (epoch + 1) % config.save_every == 0 or epoch == config.num_epochs - 1:
            rmse, mae = evaluate(basis_function, core_optim, loader, device, ind_input)
            print(f"[epoch {epoch}] eval RMSE={rmse:.5f}  MAE={mae:.5f}")
            if rmse < best_rmse:
                best_rmse = rmse
                core_path = f"./data/core_{tag}.pt"
                basis_path = f"./ckp/basis_{tag}.pth"
                torch.save(core_optim.cores, core_path)
                torch.save(basis_function.state_dict(), basis_path)
                print(f"  saved {core_path} and {basis_path}")


if __name__ == "__main__":
    main()
