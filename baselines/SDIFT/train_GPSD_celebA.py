import os
import argparse
import copy
import logging
import random
from datetime import datetime

import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm

from networks_edm import UNet


join = os.path.join


# EDM training / sampling (deterministic Algorithm 2 from the EDM paper)
@torch.no_grad()
def edm_sampler(edm, latents, t, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, use_ema=True):
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    i_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    i_steps = torch.cat([edm.round_sigma(i_steps), torch.zeros_like(i_steps[:1])])

    x_next = latents.to(torch.float64) * i_steps[0]
    for i, (i_cur, i_next) in enumerate(zip(i_steps[:-1], i_steps[1:])):
        x_hat = x_next
        denoised = edm(x_hat, i_cur, t, use_ema=use_ema).to(torch.float64)
        d_cur = (x_hat - denoised) / i_cur
        x_next = x_hat + (i_next - i_cur) * d_cur
        if i < num_steps - 1:
            denoised = edm(x_next, i_next, t, use_ema=use_ema).to(torch.float64)
            d_prime = (x_next - denoised) / i_next
            x_next = x_hat + (i_next - i_cur) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


class EDM:
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.model = model.to(self.device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max
        self.rho = cfg.rho
        self.sigma_data = cfg.sigma_data
        self.P_mean = -1.2
        self.P_std = 1.2
        self.ema_rampup_ratio = 0.05
        self.ema_halflife_kimg = 500

    def model_forward_wrapper(self, x, sigma, t, use_ema=False):
        # x: [B, T, R1=256, R2=256, R3=3], sigma: [B], t: [B, T, 1]
        # The U-Net Conv2d layers want NCHW, so we treat the size-3 R3 axis as
        # channels and permute (B, T, R1, R2, R3) -> (B, T, R3, R1, R2) before
        # the model and back after.
        sigma[sigma == 0] = self.sigma_min
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        net = self.ema if use_ema else self.model
        scaled = torch.einsum("b,btijk->btijk", c_in, x)
        scaled_nchw = scaled.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, R3, R1, R2)
        out_nchw = net(scaled_nchw, c_noise.view(-1, 1, 1).repeat(1, t.shape[1], 1), t)
        out = out_nchw.permute(0, 1, 3, 4, 2).contiguous()  # back to (B, T, R1, R2, R3)
        return torch.einsum("b,btijk->btijk", c_skip, x) + torch.einsum("b,btijk->btijk", c_out, out)

    def train_step(self, signals, t):
        rnd = torch.randn([signals.shape[0]], device=signals.device)
        sigma = (rnd * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # T=1, no GP correlation needed: noise is plain Gaussian.
        noise = torch.randn_like(signals)
        n = torch.einsum("b,btijk->btijk", sigma, noise)
        D_yn = self.model_forward_wrapper(signals + n, sigma, t)
        if self.cfg.gt_guide_type == "l2":
            loss = torch.einsum("b,btijk->btijk", weight, (D_yn - signals) ** 2)
        elif self.cfg.gt_guide_type == "l1":
            loss = torch.einsum("b,btijk->btijk", weight, torch.abs(D_yn - signals))
        else:
            raise NotImplementedError(self.cfg.gt_guide_type)
        return loss.mean()

    def update_ema(self, step, batch_size):
        ema_halflife_nimg = self.ema_halflife_kimg * 1000
        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, step * batch_size * self.ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(self.ema.parameters(), self.model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

    def __call__(self, x, sigma, t, use_ema=True):
        if sigma.shape == torch.Size([]):
            sigma = sigma * torch.ones([x.shape[0]]).to(x.device)
        return self.model_forward_wrapper(x.float(), sigma.float(), t.float(), use_ema=use_ema)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# Helpers
def create_model(cfg):
    """Plain DDPM++ U-Net (no temporal Conv1d) since T=1 for static images."""
    net = UNet(
        in_channels=cfg.channels,
        out_channels=cfg.channels,
        num_blocks=cfg.layers_per_block,
        attn_resolutions=cfg.attn_resolutions,
        model_channels=cfg.model_channels,
        channel_mult=cfg.channel_mult,
        dropout=cfg.dropout,
        img_resolution=cfg.img_size,
        label_dim=0,
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        augment_dim=9,
        channel_mult_noise=1,
        resample_filter=[1, 1],
    )
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"GPSD U-Net trainable params: {n_trainable:,}")
    return net, n_trainable


def normalize_data(x):
    mn = x.min()
    rng = x.max() - x.min()
    return (x - mn) / rng, mn, rng


def load_cores(cfg, device):
    """Load FTM cores (.pt produced by train_FTM, or .mat for back-compat).

    Cores stay on CPU (~22.6 GB for 30k CelebA-HQ samples) — they're shipped
    to the GPU one batch at a time. core_mean/core_std are scalars and live
    on GPU for use during sampling.
    """
    ext = os.path.splitext(cfg.core_path)[1].lower()
    if ext == ".pt":
        core = torch.load(cfg.core_path, map_location="cpu")
        if not isinstance(core, torch.Tensor):
            raise RuntimeError(f"unexpected pickle in {cfg.core_path}")
        core = core.float()
    elif ext == ".mat":
        d = sio.loadmat(cfg.core_path)
        core = torch.tensor(d["core"], dtype=torch.float32)
    else:
        raise ValueError(f"unknown core file extension: {ext}")
    print(f"loaded cores: {tuple(core.shape)} from {cfg.core_path}")
    core, core_mean, core_std = normalize_data(core)
    return core, core_mean.to(device), core_std.to(device)


def save_grid(samples, sample_dir, step):
    """Decode samples are still in core space here; just dump a small montage of channel-0 slices."""
    n = min(samples.shape[0], 4)
    fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axs = [axs]
    for i in range(n):
        s = samples[i, 0].detach().cpu().numpy()  # (R1, R2, R3)
        s = (s - s.min()) / max(s.max() - s.min(), 1e-8)
        axs[i].imshow(s)  # R3=3 → RGB-style preview
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(f"{sample_dir}/cores_{step}.png")
    plt.close()


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=str, default="gpsd")
    parser.add_argument("--dataset", type=str, default="celebahq")
    parser.add_argument("--core_path", type=str, required=True, help="path to cores .pt from train_FTM")
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=400_000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--save_model_iters", type=int, default=20_000)
    parser.add_argument("--save_signals_step", type=int, default=2_000)
    parser.add_argument("--log_step", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=2_000)
    # EDM
    parser.add_argument("--gt_guide_type", type=str, default="l2")
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    # Sampling
    parser.add_argument("--total_steps", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    # Architecture (defaults chosen so FTM + GPSD ≈ 134M trainable params)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--model_channels", type=int, default=128)
    parser.add_argument("--channel_mult", type=int, nargs="+", default=[1, 1, 2, 2, 4, 4])
    parser.add_argument("--attn_resolutions", type=int, nargs="+", default=[16])
    parser.add_argument("--layers_per_block", type=int, default=2,
                        help="With mc=128 mult=[1,1,2,2,4,4] this gives UNet=124.5M + FTM=3.7M ≈ 128M trainable.")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--device", type=str, default="cuda:0")
    cfg = parser.parse_args()

    cfg.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print("device:", cfg.device)

    # workdir
    cfg.expr = f"{cfg.expr}_{cfg.dataset}"
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    outdir = f"exps/{cfg.expr}_{run_id}"
    os.makedirs(outdir, exist_ok=True)
    sample_dir = join(outdir, "samples")
    ckpt_dir = join(outdir, "checkpoints")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    logging.basicConfig(
        filename=join(outdir, "std.log"),
        filemode="w",
        format="%(asctime)s %(levelname)s --> %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    logger.info("######### Arguments #########")
    for arg in vars(cfg):
        logger.info(f"  {arg}: {getattr(cfg, arg)}")

    # seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # data — cores stay on CPU, batches stream to GPU per step.
    cores, core_mean, core_std = load_cores(cfg, cfg.device)
    sio.savemat(
        join(outdir, "core_mean_std.mat"),
        {"core_mean": core_mean.cpu().numpy(), "core_std": core_std.cpu().numpy()},
    )
    t_axis = torch.linspace(0, 1, cores.shape[1]).view(1, -1, 1).repeat(cores.shape[0], 1, 1)
    dataset = torch.utils.data.TensorDataset(cores, t_axis)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=0, drop_last=True,
    )
    logger.info(f"train_loader length: {len(train_loader)}")

    # model
    net, n_trainable = create_model(cfg)
    logger.info(f"GPSD U-Net trainable params: {n_trainable:,}")
    edm = EDM(model=net, cfg=cfg)
    edm.model.train()
    optimizer = torch.optim.Adam(edm.model.parameters(), lr=cfg.learning_rate)

    progress = tqdm(total=cfg.num_steps)
    train_loss_acc = 0.0
    data_iter = iter(train_loader)
    for step in range(cfg.num_steps):
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=cfg.device)
        for _ in range(cfg.accumulation_steps):
            try:
                signal_batch, t_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                signal_batch, t_batch = next(data_iter)
            signal_batch = signal_batch.to(cfg.device, non_blocking=True)
            t_batch = t_batch.to(cfg.device, non_blocking=True)
            loss = edm.train_step(signal_batch, t_batch) / cfg.accumulation_steps
            loss.backward()
            batch_loss += loss

        for g in optimizer.param_groups:
            g["lr"] = cfg.learning_rate * min(step / max(cfg.warmup, 1), 1.0)
        for p in edm.model.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)
        optimizer.step()
        train_loss_acc += batch_loss.detach().item()
        edm.update_ema(step, cfg.train_batch_size)

        progress.update(1)
        progress.set_postfix(loss=batch_loss.item())

        if step % cfg.log_step == 0 or step == cfg.num_steps - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"step {step:08d} lr={cur_lr:.6f} avg_loss={train_loss_acc / (step + 1):.6f} batch_loss={batch_loss.item():.6f}"
            )

        if cfg.save_signals_step and (step % cfg.save_signals_step == 0 or step == cfg.num_steps - 1):
            edm.model.eval()
            shape = [cfg.eval_batch_size, cores.shape[1], cores.shape[2], cores.shape[3], cores.shape[4]]
            t_grid = torch.linspace(0, 1, shape[1], device=cfg.device).view(1, -1, 1).repeat(shape[0], 1, 1)
            x_T = torch.randn(shape, device=cfg.device)
            sample = edm_sampler(edm, x_T, t_grid, num_steps=cfg.total_steps).detach()
            sample = sample * core_std + core_mean
            save_grid(sample, sample_dir, step)
            edm.model.train()

        if cfg.save_model_iters and step > 0 and (step % cfg.save_model_iters == 0 or step == cfg.num_steps - 1):
            torch.save(edm.model.state_dict(), join(ckpt_dir, f"model_{step}.pth"))
            torch.save(edm.ema.state_dict(), join(ckpt_dir, f"ema_{step}.pth"))


if __name__ == "__main__":
    main()
