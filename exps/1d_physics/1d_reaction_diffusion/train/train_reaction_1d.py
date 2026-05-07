"""train_reaction_1d.py — Conditional DDPM training on 1D Reaction-Diffusion patch-SVD factors.

Mirrors the design in `images/our_method/train.py`:
  - Scale-only normalization (per-rank for alpha, scalar for V_hat).
  - predict_xstart=True (model predicts clean x_0 directly).
  - torch.compile + num_workers + pin_memory.
  - EMA off by default.

Conditions (1D-specific):
  - alpha_ic, V_hat_ic     (token-level context, 64 cond tokens, never noised)
  - scalar log-nu, log-rho (AdaLN signal alongside diffusion timestep)

Token layout (416 total):  [COND (64) | MAIN (352)] — only MAIN tokens are noised.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from copy import deepcopy
from time import time

import numpy as np
import torch
import yaml
import wandb
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Repo paths ─────────────────────────────────────────────────────────────
_EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "${HOME}/factor_diffusion/video")
sys.path.insert(0, _EXP)

from diffusion import create_diffusion             # noqa: E402
from models.nn import update_ema                   # noqa: E402

from dataset_reaction_1d import (                  # noqa: E402
    ReactionFactor1DDataset, reconstruct_traj,
    NX, T_TRAJ,
)
from model_reaction_1d_dit import (                # noqa: E402
    build_reaction_1d_dit,
    FLAT_MAIN, FLAT_COND,
    FLAT_ALPHA, FLAT_V_HAT,
    N_MAIN_PATCH, RANK, PATCH_DIM, N_MAIN_RANK,
)


_MILESTONE_EPOCHS = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}


# ---------------------------------------------------------------------------
# Pack / unpack helpers
# ---------------------------------------------------------------------------

def pack(batch):
    """Returns (x_flat, cond_flat, nu, rho).
        x_flat    : (B, FLAT_MAIN)  — [alpha | V_hat] normalised
        cond_flat : (B, FLAT_COND)  — [alpha_ic | V_hat_ic] normalised
        nu        : (B,) normalised log-nu
        rho       : (B,) normalised log-rho
    """
    alpha    = batch["alpha"]
    V_hat    = batch["V_hat"]
    alpha_ic = batch["alpha_ic"]
    V_hat_ic = batch["V_hat_ic"]
    nu       = batch["nu"]
    rho      = batch["rho"]

    x_flat    = torch.cat([alpha.flatten(1), V_hat.flatten(1)], dim=1)
    cond_flat = torch.cat([alpha_ic.flatten(1), V_hat_ic.flatten(1)], dim=1)
    return x_flat, cond_flat, nu, rho


def unpack_x(x_flat, B):
    c0, c1 = x_flat.split([FLAT_ALPHA, FLAT_V_HAT], dim=1)
    alpha = c0.reshape(B, N_MAIN_PATCH, RANK)
    V_hat = c1.reshape(B, PATCH_DIM, N_MAIN_RANK)
    return alpha, V_hat


def batch_to_device(batch, device):
    """Move tensor entries of a dataloader batch dict to device (non_blocking)."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Reconstruction in physical units (denormalise then invert patchify)
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_physical(alpha_norm, V_hat_norm, train_dataset, device):
    alpha = train_dataset.denorm(alpha_norm.to(device), "alpha")
    V_hat = train_dataset.denorm(V_hat_norm.to(device), "V_hat")
    return reconstruct_traj(alpha, V_hat)   # (..., 1024, 200)


# ---------------------------------------------------------------------------
# 2D heatmap visualization figure  (real | generated | residual) per row
# ---------------------------------------------------------------------------

def render_heatmap_grid(
    trajs_real: list[np.ndarray],
    trajs_gen:  list[np.ndarray],
    epoch: int, step: int, mean_rel_err: float,
):
    """Each traj is (1024, 200) numpy. Per-panel asymmetric vmin/vmax."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    B = len(trajs_real)
    fig, axes = plt.subplots(B, 3, figsize=(12, 3 * B), constrained_layout=True)
    if B == 1:
        axes = axes[None, :]

    for i in range(B):
        real = trajs_real[i]
        gen  = trajs_gen[i]
        err  = real - gen
        rel = np.linalg.norm(err) / max(np.linalg.norm(real), 1e-12)

        common = dict(aspect="auto", origin="lower",
                      extent=[0, T_TRAJ, 0, NX], cmap="RdBu_r")

        im0 = axes[i, 0].imshow(real, vmin=real.min(), vmax=real.max(), **common)
        axes[i, 0].set_title(
            f"real (sample {i})  range=[{real.min():.3g}, {real.max():.3g}]"
        )
        axes[i, 0].set_ylabel("space x")
        axes[i, 0].set_xlabel("time")
        plt.colorbar(im0, ax=axes[i, 0], shrink=0.8)

        im1 = axes[i, 1].imshow(gen, vmin=gen.min(), vmax=gen.max(), **common)
        axes[i, 1].set_title(
            f"generated  RelErr={rel:.3e}\n"
            f"range=[{gen.min():.3g}, {gen.max():.3g}]"
        )
        axes[i, 1].set_xlabel("time")
        plt.colorbar(im1, ax=axes[i, 1], shrink=0.8)

        im2 = axes[i, 2].imshow(err, vmin=err.min(), vmax=err.max(), **common)
        axes[i, 2].set_title(
            f"residual  range=[{err.min():.3g}, {err.max():.3g}]"
        )
        axes[i, 2].set_xlabel("time")
        plt.colorbar(im2, ax=axes[i, 2], shrink=0.8)

    fig.suptitle(f"epoch={epoch}  step={step}  mean rel_err={mean_rel_err:.4f}",
                 fontsize=11)
    return fig


# ---------------------------------------------------------------------------
# Visualization (every vis_every_epochs)
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_n_samples(
    wrapper, diffusion_sample, test_dataset, train_dataset, device,
    n_vis: int, step: int, epoch: int,
):
    wrapper.eval()
    B = min(n_vis, len(test_dataset))

    vis_indices = torch.randperm(len(test_dataset))[:B].tolist()
    samples = [test_dataset[i] for i in vis_indices]
    test_batch = {
        k: (torch.stack([s[k] for s in samples])
            if k != "idx" else [s[k] for s in samples])
        for k in samples[0].keys()
    }
    test_batch = batch_to_device(test_batch, device)

    x_flat, cond_flat, nu_batch, rho_batch = pack(test_batch)

    noise = torch.randn(B, FLAT_MAIN, device=device)
    samples_pred = diffusion_sample.p_sample_loop(
        wrapper, noise.shape, noise=noise,
        clip_denoised=False,
        model_kwargs={"cond_flat": cond_flat, "nu": nu_batch, "rho": rho_batch},
        device=device, progress=False,
    )

    alpha_gen,  V_hat_gen  = unpack_x(samples_pred.float(), B)
    alpha_real, V_hat_real = unpack_x(x_flat.float(),       B)

    traj_real = reconstruct_physical(alpha_real, V_hat_real, train_dataset, device).cpu().numpy()
    traj_gen  = reconstruct_physical(alpha_gen,  V_hat_gen,  train_dataset, device).cpu().numpy()

    rel_errors = [
        float(np.linalg.norm(traj_real[i] - traj_gen[i])
              / max(np.linalg.norm(traj_real[i]), 1e-8))
        for i in range(B)
    ]
    mean_rel = float(np.mean(rel_errors))
    print(f"  [vis] rel_err per sample: {[f'{e:.4f}' for e in rel_errors]}  mean={mean_rel:.4f}")

    log_dict = {"vis/rel_error_mean": mean_rel}
    fig = render_heatmap_grid(
        [traj_real[i] for i in range(B)],
        [traj_gen[i]  for i in range(B)],
        epoch, step, mean_rel,
    )
    log_dict["vis/heatmaps"] = wandb.Image(fig)

    import matplotlib.pyplot as plt
    plt.close(fig)
    wrapper.train()
    return log_dict


# ---------------------------------------------------------------------------
# Eval pass (every save_every_epochs / 100): n_eval test samples, mean RelErr
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_relerr(
    wrapper, diffusion_sample, test_dataset, train_dataset, device,
    n_eval: int, step: int, epoch: int, batch_size: int = 20,
):
    wrapper.eval()
    n = min(n_eval, len(test_dataset))
    indices = torch.randperm(len(test_dataset))[:n].tolist()

    rel_errors = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        sub = [test_dataset[i] for i in indices[s:e]]
        batch = {
            k: (torch.stack([x[k] for x in sub])
                if k != "idx" else [x[k] for x in sub])
            for k in sub[0].keys()
        }
        batch = batch_to_device(batch, device)
        x_flat, cond_flat, nu_batch, rho_batch = pack(batch)
        B = x_flat.shape[0]

        noise = torch.randn(B, FLAT_MAIN, device=device)
        pred = diffusion_sample.p_sample_loop(
            wrapper, noise.shape, noise=noise,
            clip_denoised=False,
            model_kwargs={"cond_flat": cond_flat, "nu": nu_batch, "rho": rho_batch},
            device=device, progress=False,
        )

        alpha_gen,  V_hat_gen  = unpack_x(pred.float(),    B)
        alpha_real, V_hat_real = unpack_x(x_flat.float(),  B)
        traj_real = reconstruct_physical(alpha_real, V_hat_real, train_dataset, device)
        traj_gen  = reconstruct_physical(alpha_gen,  V_hat_gen,  train_dataset, device)

        diff = (traj_real - traj_gen).reshape(B, -1)
        base = traj_real.reshape(B, -1)
        per = (diff.norm(dim=1) / base.norm(dim=1).clamp(min=1e-8)).cpu().tolist()
        rel_errors.extend(per)

    mean_rel = float(np.mean(rel_errors))
    median_rel = float(np.median(rel_errors))
    max_rel = float(np.max(rel_errors))
    print(f"  [eval n={n}] mean RelErr={mean_rel:.4e}  "
          f"median={median_rel:.4e}  max={max_rel:.4e}")
    wrapper.train()
    return {
        "eval/rel_error_mean":   mean_rel,
        "eval/rel_error_median": median_rel,
        "eval/rel_error_max":    max_rel,
        "eval/n_samples":        n,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_epoch(p):
    try:
        return int(os.path.basename(p).split("_")[0].replace("epoch", ""))
    except (ValueError, IndexError):
        return -1


def save_and_prune_checkpoints(ckpt_path, ckpt_dir, payload):
    torch.save(payload, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")
    all_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    regular = [p for p in all_ckpts if _ckpt_epoch(p) not in _MILESTONE_EPOCHS]
    for old in regular[:-1]:
        os.remove(old)
        print(f"  removed old non-milestone ckpt: {old}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train 1D Reaction-Diffusion Factor DiT")
    parser.add_argument("--config", type=str, default=None)
    # Most-overridable args
    parser.add_argument("--data_path",         type=str,   default=None)
    parser.add_argument("--test_data_path",    type=str,   default=None)
    parser.add_argument("--batch_size",        type=int,   default=None)
    parser.add_argument("--lr",                type=float, default=None)
    parser.add_argument("--n_epochs",          type=int,   default=None)
    parser.add_argument("--n_steps",           type=int,   default=None)
    parser.add_argument("--hidden_size",       type=int,   default=None)
    parser.add_argument("--depth",             type=int,   default=None)
    parser.add_argument("--num_heads",         type=int,   default=None)
    parser.add_argument("--mlp_ratio",         type=float, default=None)
    parser.add_argument("--pos_embed_2d",      type=str,   default=None)
    parser.add_argument("--log_every",         type=int,   default=None)
    parser.add_argument("--vis_every_epochs",  type=int,   default=None)
    parser.add_argument("--save_every_epochs", type=int,   default=None)
    parser.add_argument("--n_vis",             type=int,   default=None)
    parser.add_argument("--n_eval",            type=int,   default=None)
    parser.add_argument("--T_diffusion",       type=int,   default=None)
    parser.add_argument("--sample_steps",      type=int,   default=None)
    parser.add_argument("--noise_schedule",    type=str,   default=None)
    parser.add_argument("--predict_xstart",    type=str,   default=None)
    parser.add_argument("--ema",               type=str,   default=None)
    parser.add_argument("--ema_rate",          type=float, default=None)
    parser.add_argument("--grad_clip",         type=float, default=None)
    parser.add_argument("--mixed_precision",   type=str,   default=None)
    parser.add_argument("--compile",           type=str,   default=None)
    parser.add_argument("--num_workers",       type=int,   default=None)
    parser.add_argument("--prefetch_factor",   type=int,   default=None)
    parser.add_argument("--outdir",            type=str,   default=None)
    parser.add_argument("--wandb_project",     type=str,   default=None)
    parser.add_argument("--wandb_run",         type=str,   default=None)
    parser.add_argument("--wandb_entity",      type=str,   default=None)
    parser.add_argument("--wandb_mode",        type=str,   default=None)
    parser.add_argument("--device",            type=str,   default=None)
    parser.add_argument("--seed",              type=int,   default=None)
    parser.add_argument("--resume",            type=str,   default=None)
    cli = parser.parse_args()

    defaults = dict(
        data_path         = "${DATA_ROOT}/tucker_factors/reaction_1d/reaction_1d_train.pt",
        test_data_path    = "${DATA_ROOT}/tucker_factors/reaction_1d/reaction_1d_test.pt",
        batch_size        = 32,
        lr                = 1e-4,
        n_epochs          = 500,
        n_steps           = None,
        hidden_size       = 768,
        depth             = 12,
        num_heads         = 12,
        mlp_ratio         = 4.0,
        pos_embed_2d      = False,
        log_every         = 200,
        vis_every_epochs  = 10,
        save_every_epochs = 100,
        n_vis             = 4,
        n_eval            = 20,
        T_diffusion       = 1000,
        sample_steps      = 250,
        noise_schedule    = "linear",
        predict_xstart    = True,
        ema               = False,
        ema_rate          = 0.9999,
        grad_clip         = 1.0,
        mixed_precision   = True,
        compile           = True,
        num_workers       = 8,
        prefetch_factor   = 4,
        outdir            = "${DATA_ROOT}/our_method_results/reaction_1d/v3",
        wandb_project     = "<PROJECT>",
        wandb_run         = "v3_reaction1d",
        wandb_entity      = None,
        wandb_mode        = "online",
        device            = "auto",
        seed              = 0,
        resume            = None,
    )

    if cli.config is not None:
        cfg_path = cli.config if os.path.isabs(cli.config) \
            else os.path.join(_EXP, cli.config)
        with open(cfg_path) as f:
            yaml_cfg = yaml.safe_load(f)
        defaults.update({k: v for k, v in yaml_cfg.items() if v is not None})

    for key, val in vars(cli).items():
        if key == "config":
            continue
        if val is not None:
            if key in ("mixed_precision", "predict_xstart", "ema", "compile",
                       "pos_embed_2d"):
                defaults[key] = str(val).lower() != "false"
            else:
                defaults[key] = val

    args = argparse.Namespace(**defaults)

    # ── Setup ────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    ckpt_dir = os.path.join(args.outdir, "checkpoints")
    vis_dir  = os.path.join(args.outdir, "vis")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir,  exist_ok=True)

    # ── Datasets (data on CPU; stats on `device` for fast denorm) ────────
    stats_dir = os.path.join(args.outdir, "stats")
    train_dataset = ReactionFactor1DDataset(
        args.data_path, stats_dir=stats_dir, split="all", device=device,
    )
    test_dataset = ReactionFactor1DDataset(
        args.test_data_path, stats_dir=stats_dir, split="all", device=device,
        external_stats=train_dataset.stats,
    )
    print(f"Train: {len(train_dataset)}  |  Test: {len(test_dataset)}")

    n_vis  = min(args.n_vis,  len(test_dataset))
    n_eval = min(args.n_eval, len(test_dataset))

    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    # ── Model ────────────────────────────────────────────────────────────
    cfg = {
        "hidden_size":  args.hidden_size,
        "depth":        args.depth,
        "num_heads":    args.num_heads,
        "mlp_ratio":    args.mlp_ratio,
        "pos_embed_2d": args.pos_embed_2d,
    }
    wrapper = build_reaction_1d_dit(cfg).to(device)
    n_params = sum(p.numel() for p in wrapper.parameters())
    print(f"Model params: {n_params:,}  flat_main={FLAT_MAIN}  flat_cond={FLAT_COND}  "
          f"pos_embed_2d={args.pos_embed_2d}")

    if args.ema:
        ema = deepcopy(wrapper).to(device)
        ema.eval()
        update_ema(ema.parameters(), wrapper.parameters(), rate=0)
    else:
        ema = None
        print("EMA: disabled")

    # ── Diffusion (predict_xstart, learn_sigma=False) ────────────────────
    diffusion_train = create_diffusion(
        timestep_respacing="",
        noise_schedule=args.noise_schedule,
        learn_sigma=False,
        diffusion_steps=args.T_diffusion,
        predict_xstart=args.predict_xstart,
    )
    diffusion_sample = create_diffusion(
        timestep_respacing=str(args.sample_steps),
        noise_schedule=args.noise_schedule,
        learn_sigma=False,
        diffusion_steps=args.T_diffusion,
        predict_xstart=args.predict_xstart,
    )
    print(f"Diffusion: predict_xstart={args.predict_xstart}  "
          f"T={args.T_diffusion}  sample_steps={args.sample_steps}")

    # ── Optimizer ────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(wrapper.parameters(), lr=args.lr, weight_decay=0)
    use_amp = args.mixed_precision and device.type == "cuda"
    print(f"Mixed precision: {'bfloat16' if use_amp else 'off'}")

    # ── Resume (must happen before torch.compile to get clean state dicts) ──
    step = 0
    start_epoch = 0
    resume_path = args.resume
    if resume_path is None:
        existing = sorted(
            glob.glob(os.path.join(ckpt_dir, "*.pt")),
            key=lambda p: _ckpt_epoch(p),
        )
        if existing:
            resume_path = existing[-1]
            print(f"Auto-resume: {resume_path}")
    if resume_path:
        print(f"Loading checkpoint {resume_path} ...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt["model"].items()}
        wrapper.load_state_dict(sd)
        if args.ema and ckpt.get("ema") is not None:
            ema_sd = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt["ema"].items()}
            ema.load_state_dict(ema_sd)
        opt.load_state_dict(ckpt["opt"])
        step        = ckpt["step"]
        start_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed at step={step}  epoch={start_epoch}")

    # ── torch.compile ────────────────────────────────────────────────────
    if args.compile and device.type == "cuda":
        print("Compiling model with mode='reduce-overhead' ...")
        wrapper = torch.compile(wrapper, mode="reduce-overhead")

    # ── Early-exit if already at or beyond target ────────────────────────
    if args.n_epochs is not None and start_epoch >= args.n_epochs:
        print(f"Already at epoch {start_epoch} >= target {args.n_epochs}; nothing to do.")
        return

    # ── WandB ────────────────────────────────────────────────────────────
    wandb.init(
        project = args.wandb_project,
        name    = args.wandb_run,
        entity  = args.wandb_entity,
        config  = vars(args),
        dir     = args.outdir,
        mode    = args.wandb_mode,
        resume  = "allow",
    )
    wandb.config.update({"n_params": n_params}, allow_val_change=True)

    # ── Total steps ──────────────────────────────────────────────────────
    steps_per_epoch = max(len(train_dataset) // args.batch_size, 1)
    if args.n_epochs is not None:
        total_steps = args.n_epochs * steps_per_epoch
        print(f"n_epochs={args.n_epochs}  steps/epoch={steps_per_epoch}  "
              f"total_steps={total_steps}")
    else:
        total_steps = args.n_steps
        print(f"total_steps={total_steps}  "
              f"(~{total_steps / steps_per_epoch:.1f} epochs)")

    # ── Training loop ────────────────────────────────────────────────────
    wrapper.train()
    running_loss = 0.0
    log_steps    = 0
    train_start  = time()
    done         = False
    epoch        = start_epoch

    while not done:
        pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            batch = batch_to_device(batch, device)
            x_flat, cond_flat, nu_batch, rho_batch = pack(batch)

            ts = torch.randint(0, diffusion_train.num_timesteps,
                               (x_flat.shape[0],), device=device)

            with autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss_dict = diffusion_train.training_losses(
                    wrapper, x_flat, ts,
                    model_kwargs={"cond_flat": cond_flat, "nu": nu_batch, "rho": rho_batch},
                )
                loss = loss_dict["loss"].mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), args.grad_clip)
            opt.step()
            if args.ema:
                update_ema(ema.parameters(), wrapper.parameters(), rate=args.ema_rate)

            running_loss += loss.item()
            log_steps += 1
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=step, epoch=epoch)

            if step % args.log_every == 0:
                avg_loss = running_loss / log_steps
                elapsed_h = (time() - train_start) / 3600
                mem_gb = (torch.cuda.max_memory_allocated(device) / 1024 ** 3
                          if device.type == "cuda" else 0.0)
                print(f"step={step:07d}  epoch={epoch}  loss={avg_loss:.4f}  "
                      f"elapsed={elapsed_h:.3f}h  mem={mem_gb:.2f}GB")
                wandb.log({
                    "train/loss":          avg_loss,
                    "train/epoch":         epoch,
                    "train/elapsed_hours": elapsed_h,
                    "train/peak_mem_gb":   mem_gb,
                }, step=step)
                running_loss = 0.0
                log_steps = 0
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

            if args.n_epochs is None and step >= total_steps:
                done = True
                break

        epoch += 1

        is_last_target = (args.n_epochs is not None and epoch >= args.n_epochs)
        if is_last_target:
            done = True

        # Choose model used for sampling: EMA if enabled, else the live model.
        sample_model = ema if (args.ema and ema is not None) else wrapper
        # If torch.compile wrapped the model, sample on the underlying module.
        sample_model = getattr(sample_model, "_orig_mod", sample_model)

        if epoch % args.vis_every_epochs == 0 or done:
            vis_log = visualize_n_samples(
                sample_model, diffusion_sample, test_dataset, train_dataset, device,
                n_vis, step, epoch,
            )
            wandb.log({"train/epoch": epoch, **vis_log}, step=step)

        if epoch % args.save_every_epochs == 0 or done or epoch in _MILESTONE_EPOCHS:
            eval_log = eval_relerr(
                sample_model, diffusion_sample, test_dataset, train_dataset, device,
                n_eval, step, epoch,
            )
            wandb.log({"train/epoch": epoch, **eval_log}, step=step)

            ckpt_path = os.path.join(ckpt_dir, f"epoch{epoch:05d}_step{step:07d}.pt")
            raw_w = getattr(wrapper, "_orig_mod", wrapper)
            payload = {
                "model": raw_w.state_dict(),
                "ema":   (ema.state_dict() if (args.ema and ema is not None) else None),
                "opt":   opt.state_dict(),
                "step":  step,
                "epoch": epoch,
                "cfg":   cfg,
                "args":  vars(args),
            }
            save_and_prune_checkpoints(ckpt_path, ckpt_dir, payload)

    wandb.finish()
    print(f"Training complete.  steps={step}  epochs={epoch}")


if __name__ == "__main__":
    main()
