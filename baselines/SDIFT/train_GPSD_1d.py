from __future__ import annotations

import argparse
import copy
import logging
import os
import random
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

from networks_edm_1d import Spatial_temporal_UNet_1D
from utils_1d import get_gp_covariance


# EDM sampler.
@torch.no_grad()
def edm_sampler(edm, latents, t, num_steps=18, sigma_min=0.002, sigma_max=80.0,
                rho=7.0, use_ema=True, cond=None, ic_clamp=None):
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    i_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    i_steps = torch.cat([edm.round_sigma(i_steps), torch.zeros_like(i_steps[:1])])

    x_next = latents.to(torch.float64) * i_steps[0]
    if ic_clamp is not None:
        x_next[:, 0, ...] = ic_clamp.to(x_next.dtype)

    for i, (i_cur, i_next) in enumerate(zip(i_steps[:-1], i_steps[1:])):
        x_hat = x_next
        i_hat = i_cur
        denoised = edm(x_hat, i_hat, t, use_ema=use_ema, cond=cond).to(torch.float64)
        d_cur = (x_hat - denoised) / i_hat
        x_next = x_hat + (i_next - i_hat) * d_cur
        if i < num_steps - 1:
            denoised = edm(x_next, i_next, t, use_ema=use_ema, cond=cond).to(torch.float64)
            d_prime = (x_next - denoised) / i_next
            x_next = x_hat + (i_next - i_hat) * (0.5 * d_cur + 0.5 * d_prime)
        if ic_clamp is not None:
            x_next[:, 0, ...] = ic_clamp.to(x_next.dtype)
    return x_next


# EDM wrapper.
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

    def model_forward_wrapper(self, x, sigma, t, use_ema=False, cond=None):
        sigma = sigma.clone()
        sigma[sigma == 0] = self.sigma_min
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        net = self.ema if use_ema else self.model
        # x: (B, T, C, R1)
        x_in = torch.einsum("b,btcr->btcr", c_in, x)
        c_noise_per_t = c_noise.view(-1, 1, 1).repeat(1, t.shape[1], 1)
        out = net(x_in, c_noise_per_t, t, cond=cond)
        return torch.einsum("b,btcr->btcr", c_skip, x) + torch.einsum("b,btcr->btcr", c_out, out)

    def train_step(self, signals, t, cond=None, loss_mask=None, ic_clamp=None):
        rnd = torch.randn([signals.shape[0]], device=signals.device)
        sigma = (rnd * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y = signals
        cov = get_gp_covariance(t)                         # (B, T, T)
        L = torch.linalg.cholesky(cov)
        noise = torch.randn_like(y)
        n = noise.view(y.shape[0], y.shape[1], -1)
        n = (L @ n).view(y.shape)
        n = torch.einsum("b,btcr->btcr", sigma, n)

        x_in = y + n
        if ic_clamp is not None:
            x_in[:, 0, ...] = ic_clamp.to(x_in.dtype)

        D_yn = self.model_forward_wrapper(x_in, sigma, t, cond=cond)
        diff = D_yn - y
        if self.cfg.gt_guide_type == "l2":
            per = diff ** 2
        elif self.cfg.gt_guide_type == "l1":
            per = diff.abs()
        else:
            raise NotImplementedError(self.cfg.gt_guide_type)

        per = torch.einsum("b,btcr->btcr", weight, per)
        if loss_mask is not None:
            m = loss_mask.float().view(1, -1, 1, 1).to(per.device)
            per = per * m
            denom = m.sum() * per.shape[0] * per.shape[2] * per.shape[3]
            return per.sum() / denom.clamp(min=1.0)
        return per.mean()

    def update_ema(self, step, batch_size):
        ema_halflife_nimg = self.ema_halflife_kimg * 1000
        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, step * batch_size * self.ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(self.ema.parameters(), self.model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

    def __call__(self, x, sigma, t, use_ema=True, cond=None):
        if sigma.shape == torch.Size([]):
            sigma = sigma * torch.ones([x.shape[0]]).to(x.device)
        return self.model_forward_wrapper(x.float(), sigma.float(), t.float(),
                                          use_ema=use_ema, cond=cond)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# Data.
def normalize_to_unit(tr_y):
    mn = tr_y.min()
    rg = tr_y.max() - tr_y.min()
    return (tr_y - mn) / rg, mn, rg


def build_cond_vector(scalars, c0, scalar_means, scalar_stds):
    """Build the per-sample conditioning vector.

    scalars: list of (B,) tensors (log10-space already if applicable would be
        wrong here -- pass raw positive scalars; this fn does the log10 norm).
    Returns (B, len(scalars) + R1).
    """
    feats = []
    for s, m, sd in zip(scalars, scalar_means, scalar_stds):
        log_s = torch.log10(s.clamp(min=1e-12))
        feats.append(((log_s - m) / (sd + 1e-8)).unsqueeze(1))
    feats.append(c0)
    return torch.cat(feats, dim=1)


def load_train_data(cfg, device):
    d = sio.loadmat(cfg.core_path)
    core = torch.tensor(d["core"], dtype=torch.float32).to(device)   # (N, T, R1)
    nu = None
    rho = None
    if "nu" in d:
        nu = torch.tensor(np.asarray(d["nu"]).reshape(-1), dtype=torch.float32).to(device)
    if "rho" in d:
        rho = torch.tensor(np.asarray(d["rho"]).reshape(-1), dtype=torch.float32).to(device)
    if cfg.max_samples > 0 and cfg.max_samples < core.shape[0]:
        core = core[: cfg.max_samples]
        if nu is not None:
            nu = nu[: cfg.max_samples]
        if rho is not None:
            rho = rho[: cfg.max_samples]
    core, core_min, core_range = normalize_to_unit(core)
    N, T, R1 = core.shape
    print(f"core shape: {core.shape}; core_min={core_min.item():.4f}; range={core_range.item():.4f}")

    # Spatial patching: (N, T, R1) -> (N, T, patch_r1, R1//patch_r1).
    # No patch (patch_r1=1) gives the original (N, T, 1, R1).
    if cfg.patch_r1 > 1:
        assert R1 % cfg.patch_r1 == 0, f"R1 ({R1}) must be divisible by --patch_r1 ({cfg.patch_r1})"
        R1_eff = R1 // cfg.patch_r1
        core = core.reshape(N, T, cfg.patch_r1, R1_eff)
        print(f"  patched: (N, T, {cfg.patch_r1}, {R1_eff}); UNet sees in_channels={cfg.patch_r1}, R1={R1_eff}")
    else:
        core = core.unsqueeze(2)                  # (N, T, 1, R1)
    t = torch.linspace(0, 1, T, device=device).view(1, -1, 1).repeat(N, 1, 1)

    if cfg.conditional:
        assert nu is not None, "core .mat is missing 'nu' field; re-run FTM training."
        scalars = [nu]
        scalar_names = ["nu"]
        if rho is not None:
            scalars.append(rho)
            scalar_names.append("rho")
        means, stds = [], []
        for s in scalars:
            ls = torch.log10(s.clamp(min=1e-12))
            means.append(ls.mean())
            stds.append(ls.std())
        # IC slice: flatten to original R1 for the cond MLP (preserve patch reshape underneath).
        c0 = core[:, 0, ...].reshape(N, R1).clone()           # (N, R1)
        cond = build_cond_vector(scalars, c0, means, stds)
        ds = torch.utils.data.TensorDataset(core, t, cond)
        cond_meta = dict(
            cond_dim=int(cond.shape[1]),
            scalar_names=",".join(scalar_names),
            log_nu_mean=float(means[0].item()),
            log_nu_std=float(stds[0].item()),
        )
        if "rho" in scalar_names:
            cond_meta["log_rho_mean"] = float(means[1].item())
            cond_meta["log_rho_std"] = float(stds[1].item())
        print(f"conditioning scalars: {scalar_names}; cond_dim={cond.shape[1]}")
    else:
        ds = torch.utils.data.TensorDataset(core, t)
        cond_meta = None

    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.train_batch_size, shuffle=True, num_workers=0, drop_last=False
    )
    return loader, core_min, core_range, cond_meta


# Main.
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--expr", default="gp-edm")
    p.add_argument("--dataset", default="burgers_1d")
    p.add_argument("--core_path", required=True)
    p.add_argument("--seed", type=int, default=231)
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--num_steps", type=int, default=20001)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--save_model_iters", type=int, default=5000)
    p.add_argument("--save_signals_step", type=int, default=2000)
    p.add_argument("--log_step", type=int, default=200)
    # EDM.
    p.add_argument("--gt_guide_type", default="l2")
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--sigma_data", type=float, default=0.5)
    p.add_argument("--total_steps", type=int, default=20)
    # Architecture.
    p.add_argument("--r1_resolution", type=int, default=64)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--model_channels", type=int, default=64)
    p.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 2, 2])
    p.add_argument("--attn_resolutions", type=int, nargs="+", default=[16, 8])
    p.add_argument("--layers_per_block", type=int, default=2)
    p.add_argument("--num_temporal_latent", type=int, default=4)
    # Conditional.
    p.add_argument("--conditional", action="store_true")
    p.add_argument("--mask_t0_loss", action="store_true",
                   help="If conditional: zero out the loss on the t=0 slot (recommended).")
    # Smoke-test.
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--out_root", default="./exps")
    # Spatial patching: reshape (B, T, 1, R1) -> (B, T, patch_r1, R1//patch_r1).
    # Folds patch_r1 spatial positions into channels; UNet operates on R1//patch_r1 length.
    # No info loss; ~patch_r1 x speedup. Set --in_channels and --r1_resolution accordingly.
    p.add_argument("--patch_r1", type=int, default=1, help="Spatial patch factor (1 = no patching).")
    # Speed knobs (no architecture change).
    p.add_argument("--bf16", action="store_true",
                   help="Run forward+backward under torch.autocast(bfloat16). H100 BF16 is fast and lossless for this workload.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the denoiser model. ~1.3-1.6x speedup after warmup.")
    p.add_argument("--tf32", action="store_true", default=True,
                   help="Enable TF32 matmul + cuDNN benchmark autotuner.")
    return p.parse_args()


def main():
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device
    print("device:", device)

    # Free speed knobs (lossless on H100 for this workload).
    if cfg.tf32 and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")          # TF32 for matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True               # autotune Conv kernels
        print("speed: TF32 + cuDNN benchmark enabled")
    if cfg.bf16:
        print("speed: BF16 autocast enabled")
    if cfg.compile:
        print("speed: torch.compile enabled (warmup is slow)")

    suffix = "_cond" if cfg.conditional else "_uncond"
    cfg.expr = f"{cfg.expr}_{cfg.dataset}{suffix}"
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    outdir = f"{cfg.out_root}/{cfg.expr}_{run_id}"
    os.makedirs(outdir, exist_ok=True)
    sample_dir = f"{outdir}/samples"
    ckpt_dir = f"{outdir}/checkpoints"
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logging.basicConfig(filename=f"{outdir}/std.log", filemode="w",
                        format="%(asctime)s %(levelname)s --> %(message)s",
                        level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()
    for k, v in vars(cfg).items():
        logger.info(f"\t{k}: {v}")
    print("outdir:", outdir)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    train_loader, core_min, core_range, cond_meta = load_train_data(cfg, device)
    sio.savemat(f"{outdir}/core_norm.mat",
                {"core_min": core_min.detach().cpu().numpy(),
                 "core_range": core_range.detach().cpu().numpy()})
    if cond_meta is not None:
        sio.savemat(f"{outdir}/cond_meta.mat", cond_meta)

    cond_dim = cond_meta["cond_dim"] if cond_meta is not None else 0
    # Number of scalar conditioners that prefix the c0 slice in the cond vector.
    # cond layout: [scalar_1, ..., scalar_K, c0_0, ..., c0_{R1_data-1}]
    R1_data = cfg.patch_r1 * cfg.r1_resolution
    n_scalars = (cond_dim - R1_data) if cond_meta is not None else 0

    net = Spatial_temporal_UNet_1D(
        r1_resolution=cfg.r1_resolution,
        in_channels=cfg.channels,
        out_channels=cfg.channels,
        model_channels=cfg.model_channels,
        channel_mult=cfg.channel_mult,
        num_blocks=cfg.layers_per_block,
        num_temporal_latent=cfg.num_temporal_latent,
        attn_resolutions=cfg.attn_resolutions,
        dropout=0.0,
        cond_dim=cond_dim,
    ).to(device)
    edm = EDM(model=net, cfg=cfg)
    edm.model.train()

    n_par = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_par}")
    logger.info(f"trainable parameters: {n_par}")

    # torch.compile after EDM construction (so EMA is also built from the compiled forward).
    if cfg.compile:
        edm.model = torch.compile(edm.model, mode="reduce-overhead", fullgraph=False)
        # ema is a deepcopy of the (uncompiled) model; recompile too for inference parity
        edm.ema = torch.compile(edm.ema, mode="reduce-overhead", fullgraph=False)

    optimiser = torch.optim.Adam(edm.model.parameters(), lr=cfg.learning_rate)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (cfg.bf16 and torch.cuda.is_available())
        else torch.autocast(device_type="cpu", enabled=False)
    )

    T = train_loader.dataset.tensors[0].shape[1]
    loss_mask = None
    if cfg.conditional and cfg.mask_t0_loss:
        loss_mask = torch.ones(T, device=device)
        loss_mask[0] = 0.0

    pbar = tqdm(total=cfg.num_steps)
    train_loss_acc = 0.0
    data_iter = None
    for step in range(cfg.num_steps):
        optimiser.zero_grad(set_to_none=True)
        batch_loss = torch.tensor(0.0, device=device)
        for _ in range(cfg.accumulation_steps):
            try:
                batch = next(data_iter)
            except (StopIteration, TypeError):
                data_iter = iter(train_loader)
                batch = next(data_iter)
            if cfg.conditional:
                signals, t_b, cond_b = batch
                # IC clamp uses the c_0 slice (length = original R1 = patch_r1 * r1_resolution).
                R1_data = cfg.patch_r1 * cfg.r1_resolution
                c0_flat = cond_b[:, n_scalars: n_scalars + R1_data].to(signals.device)  # (B, R1_data)
                ic_clamp = c0_flat.view(c0_flat.shape[0], cfg.patch_r1, cfg.r1_resolution)  # (B, patch_r1, R1_eff)
            else:
                signals, t_b = batch
                cond_b = None
                ic_clamp = None
            with autocast_ctx:
                loss = edm.train_step(signals, t_b, cond=cond_b, loss_mask=loss_mask,
                                      ic_clamp=ic_clamp) / cfg.accumulation_steps
            # BF16 has enough range that GradScaler is unnecessary; backprop directly.
            loss.backward()
            batch_loss = batch_loss + loss.detach()
        for g in optimiser.param_groups:
            g["lr"] = cfg.learning_rate * min(step / max(cfg.warmup, 1), 1.0)
        for p_ in net.parameters():
            if p_.grad is not None:
                torch.nan_to_num(p_.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=p_.grad)
        optimiser.step()
        edm.update_ema(step, cfg.train_batch_size)

        # Only sync the loss to host every log_step (avoid per-step CUDA sync).
        is_log_step = (step % cfg.log_step == 0) or (step == cfg.num_steps - 1)
        if is_log_step:
            batch_loss_val = float(batch_loss.detach())
            train_loss_acc += batch_loss_val * cfg.log_step  # rough running sum
            cur_lr = optimiser.param_groups[0]["lr"]
            logger.info(f"step {step:08d} | lr {cur_lr:.6f} | avg {train_loss_acc / (step + 1):.6f} | batch {batch_loss_val:.6f}")
            pbar.set_postfix(loss=batch_loss_val)
        pbar.update(1)

        if cfg.save_signals_step and (step % cfg.save_signals_step == 0 or step == cfg.num_steps - 1):
            edm.model.eval()
            B_s = 5
            sample_shape = [B_s, T, cfg.channels, cfg.r1_resolution]
            t_grid = torch.linspace(0, 1, T, device=device).view(1, -1, 1).repeat(B_s, 1, 1)
            cov = get_gp_covariance(t_grid)
            L = torch.linalg.cholesky(cov)
            noise = torch.randn(sample_shape, device=device)
            x_T = (L @ noise.view(B_s, T, -1)).view(sample_shape)
            cond_eval = None
            ic_eval = None
            if cfg.conditional:
                cond_eval = train_loader.dataset.tensors[2][:B_s].to(device)
                _c0 = cond_eval[:, n_scalars: n_scalars + R1_data]
                ic_eval = _c0.view(B_s, cfg.patch_r1, cfg.r1_resolution)
            with torch.no_grad(), autocast_ctx:
                samp = edm_sampler(edm, x_T, t_grid, num_steps=cfg.total_steps,
                                   cond=cond_eval, ic_clamp=ic_eval).detach().cpu()
            if step >= cfg.save_model_iters:
                sio.savemat(f"{sample_dir}/core_{step}.mat", {"core": samp.numpy()})
            plt.figure(figsize=(7, 4))
            for b in range(B_s):
                plt.plot(samp[b, :, 0, 0].numpy(), color=f"C{b}", alpha=0.7,
                         label=f"sample {b}" if step == 0 else None)
            plt.title(f"r=0 trajectories @ step {step}")
            plt.xlabel("t-index"); plt.ylabel("core")
            plt.tight_layout()
            plt.savefig(f"{sample_dir}/samples_{step}.png")
            plt.close()
            edm.model.train()

        if cfg.save_model_iters and (step % cfg.save_model_iters == 0 or step == cfg.num_steps - 1) and step > 0:
            # Save EMA weights (what inference uses). Unwrap torch.compile if present.
            sd_ema = edm.ema._orig_mod.state_dict() if hasattr(edm.ema, "_orig_mod") else edm.ema.state_dict()
            torch.save(sd_ema, f"{ckpt_dir}/ema_{step}.pth")
            # Also save raw model weights for resuming training.
            sd_raw = edm.model._orig_mod.state_dict() if hasattr(edm.model, "_orig_mod") else edm.model.state_dict()
            torch.save(sd_raw, f"{ckpt_dir}/model_{step}.pth")


if __name__ == "__main__":
    main()
