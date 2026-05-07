import argparse
import copy
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

from networks_edm import Spatial_temporal_UNet



def get_gp_covariance(t, gp_gamma=50.0):
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5
    return torch.exp(-torch.square(s) * gp_gamma) + diag


@torch.no_grad()
def edm_sampler(edm, latents, t, cond=None, num_steps=18, sigma_min=0.002, sigma_max=80,
                rho=7, use_ema=True):
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64,
                                device=latents.device)
    i_steps = (sigma_max ** (1 / rho) +
               step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    i_steps = torch.cat([edm.round_sigma(i_steps), torch.zeros_like(i_steps[:1])])
    x_next = latents.to(torch.float64) * i_steps[0]
    for i, (i_cur, i_next) in enumerate(zip(i_steps[:-1], i_steps[1:])):
        x_hat = x_next
        i_hat = i_cur
        denoised = edm(x_hat, i_hat, t, cond=cond, use_ema=use_ema).to(torch.float64)
        d_cur = (x_hat - denoised) / i_hat
        x_next = x_hat + (i_next - i_hat) * d_cur
        if i < num_steps - 1:
            denoised = edm(x_next, i_next, t, cond=cond, use_ema=use_ema).to(torch.float64)
            d_prime = (x_next - denoised) / i_next
            x_next = x_hat + (i_next - i_hat) * (0.5 * d_cur + 0.5 * d_prime)
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
        self.P_mean = cfg.p_mean
        self.P_std = cfg.p_std
        self.ema_rampup_ratio = 0.05
        self.ema_halflife_kimg = 500

    def model_forward_wrapper(self, x, sigma, t, cond=None, use_ema=False):
        sigma = sigma.clone()
        sigma[sigma == 0] = self.sigma_min
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        net = self.ema if use_ema else self.model
        c_noise_b = c_noise.view(-1, 1, 1).repeat(1, t.shape[1], 1)
        x_scaled = torch.einsum('b,btijk->btijk', c_in, x)
        if cond is not None:
            # Channel-concat conditioning: (B, T, C+Ccond, H, W)
            net_input = torch.cat([x_scaled, cond.to(x_scaled.dtype)], dim=2)
        else:
            net_input = x_scaled
        model_output = net(net_input, c_noise_b, t)
        return (torch.einsum('b,btijk->btijk', c_skip, x)
                + torch.einsum('b,btijk->btijk', c_out, model_output))

    def train_step(self, signals, t):
        rnd = torch.randn([signals.shape[0]], device=signals.device)
        sigma = (rnd * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        cov = get_gp_covariance(t)
        L = torch.linalg.cholesky(cov)
        noise = torch.randn_like(signals)
        noise = L @ noise.view(signals.shape[0], signals.shape[1], -1)
        noise = noise.view(signals.shape)
        n = torch.einsum('b,btijk->btijk', sigma, noise)

        cond = None
        if getattr(self.cfg, 'cond_t0', False):
            # Clean t=0 core, broadcast across all T as a static conditioning channel
            cond = signals[:, 0:1, :, :, :].expand(-1, signals.shape[1], -1, -1, -1)

        D_yn = self.model_forward_wrapper(signals + n, sigma, t, cond=cond)

        if self.cfg.gt_guide_type == 'l2':
            err = (D_yn - signals) ** 2
        else:
            err = torch.abs(D_yn - signals)
        if cond is not None:
            # Mask out t=0 — the model trivially gets it from the cond channel
            err = err.clone()
            err[:, 0] = 0
        loss = torch.einsum('b,btijk->btijk', weight, err)
        return loss.mean()

    def update_ema(self, step, batch_size):
        ema_halflife_nimg = self.ema_halflife_kimg * 1000
        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg,
                                    step * batch_size * self.ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(self.ema.parameters(), self.model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

    def __call__(self, x, sigma, t, cond=None, use_ema=True):
        if sigma.shape == torch.Size([]):
            sigma = sigma * torch.ones([x.shape[0]]).to(x.device)
        return self.model_forward_wrapper(x.float(), sigma.float(), t.float(),
                                          cond=cond, use_ema=use_ema)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)



def normalize_data(tr):
    mn = tr.min()
    sd = tr.max() - tr.min()
    return (tr - mn) / sd, mn, sd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--core_path", type=str, required=True,
                   help="Path to FTM-produced core_best.mat (or core_epoch_NNN.mat)")
    p.add_argument("--out_dir", type=str,
                   default="/projects/p32954/bkx8728/karman_sdift_runs")
    p.add_argument("--data_name", type=str, default="karman_2d")
    p.add_argument("--seed", type=int, default=231)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Cap total steps (e.g. 50 for pilot timing).")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--accumulation_steps", type=int, default=1)
    # Architecture
    p.add_argument("--img_size", type=int, default=12)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--model_channels", type=int, default=72)
    p.add_argument("--channel_mult", type=int, nargs='+', default=[1, 2, 2])
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--num_temporal_latent", type=int, default=8)
    p.add_argument("--attn_resolutions", type=int, nargs='*', default=[])
    # EDM
    p.add_argument("--gt_guide_type", type=str, default='l2')
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--sigma_data", type=float, default=0.5)
    p.add_argument("--p_mean", type=float, default=-1.2,
                   help="Mean of log-normal noise level distribution during training.")
    p.add_argument("--p_std", type=float, default=1.2,
                   help="Std of log-normal noise level distribution during training.")
    # Logging
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every_epoch", type=int, default=25)
    p.add_argument("--sample_every_epoch", type=int, default=10)
    p.add_argument("--sample_steps", type=int, default=250,
                   help="Number of denoising iterations during EDM sampling. "
                        "Each iteration is 2 NFE (Heun 2nd order, except final) "
                        "so total model evals per sample = 2*sample_steps - 1.")
    # Speedups
    p.add_argument("--bf16", action="store_true",
                   help="Run forward/backward under torch.autocast(bfloat16). "
                        "Halves activation memory and ~1.7x faster on H100.")
    p.add_argument("--group_t", type=int, default=1,
                   help="Group K consecutive timesteps into channels. Reshape "
                        "(B, T, C, H, W) -> (B, T/K, K*C, H, W). Reduces UNet "
                        "FLOPs by ~K with minimal param impact. T must be "
                        "divisible by K (drops the tail otherwise).")
    p.add_argument("--compile", action="store_true",
                   help="Apply torch.compile to the UNet (extra ~1.3x speedup, "
                        "first epoch is slow due to compilation).")
    p.add_argument("--cond_t0", action=argparse.BooleanOptionalAction, default=False,
                   help="Channel-concat conditioning on the clean t=0 core. "
                        "When on, UNet input has 2*channels (noisy + cond_t0); "
                        "loss is masked at t=0. Pass --cond_t0 to enable.")
    return p.parse_args()


def main():
    config = parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    config.device = device

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(config.out_dir) / f"gpsd_{config.data_name}_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    sample_dir = outdir / "samples"
    ckpt_dir = outdir / "checkpoints"
    sample_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    logging.basicConfig(filename=f'{outdir}/std.log', filemode='w',
                        format='%(asctime)s %(levelname)s --> %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    for arg in vars(config):
        logger.info(f"\t{arg}: {getattr(config, arg)}")
    print(f"[device] {device}\n[outdir] {outdir}", flush=True)

    # ---- Load cores ----
    d = sio.loadmat(config.core_path)
    core = torch.tensor(d['core'], dtype=torch.float32)  # (N, T, R1, R2, R3)
    print(f"[data] cores loaded: shape={tuple(core.shape)}  "
          f"min={core.min():.3f}  max={core.max():.3f}", flush=True)
    core, c_mean, c_std = normalize_data(core)
    sio.savemat(f"{outdir}/core_mean_std.mat",
                {"core_mean": c_mean.numpy(), "core_std": c_std.numpy()})

    # ---- Optional T grouping (fold K timesteps into channels) ----
    K = max(1, int(config.group_t))
    T_full = core.shape[1]
    C_orig = core.shape[2]
    if K > 1:
        T_eff = T_full // K
        core = core[:, :T_eff * K]                      # drop tail if needed
        # (N, T_eff*K, C, H, W) -> (N, T_eff, K*C, H, W)
        core = core.reshape(core.size(0), T_eff, K, C_orig, core.size(3), core.size(4))
        core = core.reshape(core.size(0), T_eff, K * C_orig, core.size(4), core.size(5))
        print(f"[group_t] K={K}: T={T_full} -> T_eff={T_eff}, "
              f"channels={C_orig} -> {K * C_orig}", flush=True)
        config.channels = K * C_orig                    # UNet sees K*C channels
    else:
        T_eff = T_full

    core = core.to(device)
    t_grid = (torch.linspace(0, 1, T_eff).view(1, -1, 1).to(device)
              .repeat(core.shape[0], 1, 1))

    dataset = torch.utils.data.TensorDataset(core, t_grid)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    steps_per_epoch = len(train_loader)
    total_steps = config.epochs * steps_per_epoch
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)
    print(f"[data] N={core.shape[0]}  steps/epoch={steps_per_epoch}  "
          f"total_steps={total_steps}", flush=True)

    # ---- Model ----
    in_ch = (config.channels * 2) if config.cond_t0 else config.channels
    unet = Spatial_temporal_UNet(
        in_channels=in_ch,
        out_channels=config.channels,
        num_blocks=config.num_blocks,
        num_temporal_latent=config.num_temporal_latent,
        attn_resolutions=config.attn_resolutions,
        model_channels=config.model_channels,
        channel_mult=config.channel_mult,
        dropout=0,
        img_resolution=config.img_size,
        label_dim=0,
        embedding_type='positional',
        encoder_type='standard',
        decoder_type='standard',
        augment_dim=9,
        channel_mult_noise=1,
        resample_filter=[1, 1],
    )
    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"[model] GPSD params: {n_params:,}", flush=True)
    edm = EDM(model=unet, cfg=config)
    if getattr(config, "compile", False):
        try:
            edm.model = torch.compile(edm.model)
            edm.ema = torch.compile(edm.ema)
            print("[model] torch.compile applied", flush=True)
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}", flush=True)
    edm.model.train()
    optimizer = torch.optim.Adam(edm.model.parameters(), lr=config.lr)

    # ---- Train ----
    step = 0
    train_loss_avg = 0.0
    epoch_times = []
    t_start = time.time()

    for epoch in range(1, config.epochs + 1):
        ep_t0 = time.time()
        step_times = []
        amp_dtype = torch.bfloat16 if config.bf16 else None
        for signal_batch, t_batch in train_loader:
            if config.max_steps is not None and step >= config.max_steps:
                break
            step_t0 = time.time()
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            for _ in range(config.accumulation_steps):
                if amp_dtype is not None:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        loss = edm.train_step(signal_batch, t_batch)
                else:
                    loss = edm.train_step(signal_batch, t_batch)
                loss = loss / config.accumulation_steps
                loss.backward()
                batch_loss += loss
            for g in optimizer.param_groups:
                g['lr'] = config.lr * min(step / max(config.warmup, 1), 1)
            for p in unet.parameters():
                if p.grad is not None:
                    torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5,
                                     out=p.grad)
            optimizer.step()
            edm.update_ema(step + 1, config.batch_size)
            step += 1
            train_loss_avg += batch_loss.detach().item()
            step_times.append(time.time() - step_t0)

            if step % config.log_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                mean_ms = float(np.mean(step_times[-config.log_every:])) * 1000
                msg = (f"[step {step:6d}] epoch={epoch} lr={lr_now:.6f} "
                       f"loss={batch_loss.item():.5f} "
                       f"avg_loss={train_loss_avg/step:.5f} "
                       f"step_ms={mean_ms:.0f}")
                print(msg, flush=True)
                logger.info(msg)

        ep_t = time.time() - ep_t0
        epoch_times.append(ep_t)
        ep_step_ms = float(np.mean(step_times)) * 1000 if step_times else 0
        print(f"[epoch {epoch:3d}] time={ep_t:.1f}s  mean_step={ep_step_ms:.0f}ms  "
              f"step={step}", flush=True)

        if epoch % config.save_every_epoch == 0 or epoch == config.epochs:
            torch.save(edm.model.state_dict(),
                       f"{ckpt_dir}/ema_epoch_{epoch:03d}.pth")

        if epoch % config.sample_every_epoch == 0 or epoch == config.epochs:
            edm.model.eval()
            sample_shape = [4, core.shape[1], config.channels,
                            config.img_size, config.img_size]
            t_g = (torch.linspace(0, 1, sample_shape[1]).view(1, -1, 1).to(device)
                   .repeat(sample_shape[0], 1, 1))
            cov = get_gp_covariance(t_g)
            L = torch.linalg.cholesky(cov).to(device)
            noise = torch.randn(sample_shape).to(device)
            x_T = (L @ noise.view(sample_shape[0], sample_shape[1], -1)
                   ).view(sample_shape)
            sample_cond = None
            if config.cond_t0:
                # Use first 4 train cores' t=0 as conditioning for sanity-check samples
                sample_cond = (core[:4, 0:1].expand(-1, sample_shape[1], -1, -1, -1)
                               .contiguous())
            sample = edm_sampler(edm, x_T, t_g, cond=sample_cond,
                                  num_steps=config.sample_steps).detach()
            sample_unnorm = (sample * c_std.to(device) + c_mean.to(device)).cpu()
            # Ungroup time axis if K > 1: (B, T_eff, K*C, H, W) -> (B, T_eff*K, C, H, W)
            if K > 1:
                bs_, te_, kc_, h_, w_ = sample_unnorm.shape
                sample_unnorm = sample_unnorm.view(bs_, te_, K, kc_ // K, h_, w_)
                sample_unnorm = sample_unnorm.reshape(bs_, te_ * K, kc_ // K, h_, w_)
            sio.savemat(f"{sample_dir}/cores_epoch_{epoch:03d}.mat",
                        {"core": sample_unnorm.numpy()})
            edm.model.train()

        if config.max_steps is not None and step >= config.max_steps:
            print(f"[done] hit max_steps={config.max_steps}", flush=True)
            break

    total_t = time.time() - t_start
    print(f"[done] total_steps={step}  total_time={total_t:.1f}s "
          f"mean_epoch_time={np.mean(epoch_times) if epoch_times else 0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
