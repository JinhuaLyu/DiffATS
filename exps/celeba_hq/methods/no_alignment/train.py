"""
Train JointDiT on no_alignment shards.

Same model architecture as our_method (JointDiT alpha+V_hat), same data tensor
shape (B, 3, alpha_n+patch_dim, R), but the data uses per-image SVD with NO
Procrustes alignment. For preview reconstruction we load the dataset-wide
average per-image mean from mean_avg_no_alignment_p32_r32.pt.

Checkpoints + W&B sample previews fire on EPOCH boundaries.

If --ortho-augment is True, a random RxR orthogonal Q is sampled per image
and applied as (alpha, V_hat) -> (alpha @ Q, V_hat @ Q). This is the
data_augmentation training run.
"""

import os
import sys
import io
import glob
import math
import argparse
import logging
from time import time
from copy import deepcopy
from collections import OrderedDict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm
import wandb
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.amp import autocast

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.benchmark = True

# Re-use diffusion package at celeba_hq root (two levels up from methods/<name>/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from diffusion import create_diffusion

# Re-use JointDiT, ShardedProcAlphaDataset and helpers from our_method/train.py.
# Load it by explicit path under a unique name to avoid local train.py shadowing.
import importlib.util  # noqa: E402
_OUR_METHOD = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "our_method")
)
sys.path.insert(0, _OUR_METHOD)
from dit_models import JointDiT  # noqa: E402
_om_spec = importlib.util.spec_from_file_location("our_method_train", os.path.join(_OUR_METHOD, "train.py"))
_om_train = importlib.util.module_from_spec(_om_spec)
sys.modules["our_method_train"] = _om_train
_om_spec.loader.exec_module(_om_train)
ShardedProcAlphaDataset       = _om_train.ShardedProcAlphaDataset
requires_grad                 = _om_train.requires_grad
update_ema                    = _om_train.update_ema
recover_from_alpha_to_image   = _om_train.recover_from_alpha_to_image
cleanup                       = _om_train.cleanup
create_logger                 = _om_train.create_logger
infer_joint_layout            = _om_train.infer_joint_layout


def _strip_compile_prefix(sd: dict) -> dict:
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level YAML must be a mapping; got {type(cfg).__name__}")
    return cfg


def main(args):
    assert torch.cuda.is_available(), "Training requires a GPU."

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob.glob(f"{args.results_dir}/*"))
    experiment_dir   = f"{args.results_dir}/{experiment_index:03d}-JointDiT"
    checkpoint_dir   = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment dir: {experiment_dir}")
    logger.info(f"ortho_augment={args.ortho_augment}")

    # ---- Stats + preview mean ----
    stats = torch.load(args.alpha_stats_path, map_location="cpu", weights_only=False)
    alpha_rank_std = stats["std"].float()                        # (3, R)
    logger.info(f"alpha stats: std {tuple(alpha_rank_std.shape)}, "
                f"range [{alpha_rank_std.min():.4f}, {alpha_rank_std.max():.4f}]")

    vhat_ckpt = torch.load(args.vhat_stats_path, map_location="cpu", weights_only=False)
    vhat_std  = vhat_ckpt["std"].float()
    logger.info(f"V_hat std: {vhat_std:.6f}")

    mean_obj = torch.load(args.mean_avg_path, map_location="cpu", weights_only=False)
    preview_mean = mean_obj["mean_avg"].float()                  # (3, patch_dim)
    logger.info(f"preview mean (mean_avg): {tuple(preview_mean.shape)}")

    # ---- Layout ----
    layout = infer_joint_layout(args.input_size, args.patch_size, args.svd_rank)
    img_hw   = layout["img_hw"]
    patch    = layout["patch"]
    rank     = layout["rank"]
    alpha_n  = layout["alpha_n"]
    patch_dim = layout["patch_dim"]
    logger.info(f"layout: img={img_hw} patch={patch} rank={rank} alpha_n={alpha_n} patch_dim={patch_dim}")

    # ---- Dataset ----
    dataset = ShardedProcAlphaDataset(args.shard_dir, preload=False)
    assert len(dataset) >= 100
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size,
        shuffle=True,
        pin_memory=True,
        pin_memory_device="cuda",
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=False,
    )
    logger.info(f"dataset={len(dataset)} batches/epoch={len(loader)}")

    device = torch.device("cuda")

    # ---- Model + diffusion ----
    model = JointDiT(
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        img_size=img_hw,
        patch_size=patch,
        rank=rank,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume
    start_epoch = 0
    train_steps = 0
    resume_ckpt = None

    def _find_latest_ckpt(p):
        if os.path.isdir(p):
            cands = sorted(glob.glob(os.path.join(p, "*.pt")))
            if not cands:
                raise FileNotFoundError(f"No .pt in {p}")
            return cands[-1]
        return p

    if args.resume:
        ckpt_path = _find_latest_ckpt(args.resume)
        logger.info(f"[Resume] loading {ckpt_path}")
        resume_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(_strip_compile_prefix(resume_ckpt["model"]), strict=True)
        opt.load_state_dict(resume_ckpt["opt"])
        start_epoch = int(resume_ckpt.get("epoch", 0))
        train_steps = int(resume_ckpt.get("train_steps", 0))

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    if resume_ckpt is not None and resume_ckpt.get("ema") is not None:
        ema.load_state_dict(_strip_compile_prefix(resume_ckpt["ema"]), strict=True)
    del resume_ckpt

    model = torch.compile(model, mode="reduce-overhead")
    model = model.to(memory_format=torch.channels_last)

    diffusion = create_diffusion(
        learn_sigma=args.learn_sigma, timestep_respacing="", predict_xstart=True,
    )
    fast_diffusion = create_diffusion(
        learn_sigma=args.learn_sigma, timestep_respacing=str(args.sample_steps), predict_xstart=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"JointDiT params: {n_params:,} ({n_params/1e6:.2f} M)")
    logger.info(f"args: {args}")

    if args.ema:
        update_ema(ema, model, decay=0)
    model.train(); ema.eval()

    method = "data_augmentation" if args.ortho_augment else "no_alignment"
    auto_name = (
        f"{method}_JointDiT_lr{args.lr}_p{patch}_r{rank}_bs{args.global_batch_size}"
    )
    name = args.wandb_run_name or auto_name
    wandb.init(
        entity=args.wandb_entity or None,
        project=args.wandb_project,
        name=name,
        config={
            "method": method,
            "model": "JointDiT",
            "ortho_augment": args.ortho_augment,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "global_batch_size": args.global_batch_size,
            "svd_rank": rank,
            "patch": patch,
            "image_size": img_hw,
            "hidden_size": args.hidden_size,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "ckpt_every_epoch": args.ckpt_every_epoch,
            "sample_steps": args.sample_steps,
            "n_params_M": n_params / 1e6,
        },
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/loss_epoch", step_metric="epoch")
    wandb.define_metric("train/time_epoch", step_metric="epoch")
    wandb.define_metric("samples/epoch_preview", step_metric="epoch")

    log_steps = 0
    running_loss = 0.0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs (start_epoch={start_epoch}, train_steps={train_steps})")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"=== epoch {epoch} ===")
        epoch_loss_sum = torch.zeros(1, device=device)
        epoch_batches  = 0
        epoch_t0       = time()

        for rdata in tqdm(loader, desc=f"ep{epoch}"):
            alpha_raw, V_hat_raw = rdata[0].to(device), rdata[1].to(device)

            alpha_norm = alpha_raw / alpha_rank_std.to(device)[None, :, None, :]
            V_hat_norm = V_hat_raw / vhat_std.to(device)

            if args.ortho_augment:
                B_aug = alpha_norm.shape[0]
                R_aug = alpha_norm.shape[3]
                Q, _ = torch.linalg.qr(torch.randn(B_aug, R_aug, R_aug, device=device))
                N_a = alpha_norm.shape[2]
                N_v = V_hat_norm.shape[2]
                alpha_norm = torch.bmm(alpha_norm.reshape(B_aug, -1, R_aug), Q
                                       ).reshape(B_aug, 3, N_a, R_aug)
                V_hat_norm = torch.bmm(V_hat_norm.reshape(B_aug, -1, R_aug), Q
                                       ).reshape(B_aug, 3, N_v, R_aug)

            data = torch.cat([alpha_norm, V_hat_norm], dim=2)
            data = data.to(memory_format=torch.channels_last)

            t = torch.randint(0, diffusion.num_timesteps, (data.shape[0],), device=device)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(model, data, t, model_kwargs={})
                loss = loss_dict["loss"].mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if args.ema:
                update_ema(ema, model)

            running_loss   += loss.item()
            epoch_loss_sum += loss.detach()
            log_steps   += 1
            train_steps += 1
            epoch_batches += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg = running_loss / log_steps
                logger.info(f"(step={train_steps:07d}) loss={avg:.4f} steps/sec={steps_per_sec:.2f}")
                wandb.log({
                    "train/loss":          avg,
                    "train/steps_per_sec": steps_per_sec,
                    "train/step":          train_steps,
                }, commit=False)
                running_loss = 0.0
                log_steps    = 0
                start_time   = time()

        epoch_avg = (epoch_loss_sum / max(1, epoch_batches)).item()
        epoch_dur = time() - epoch_t0
        wandb.log({
            "epoch":            epoch,
            "train/loss_epoch": epoch_avg,
            "train/time_epoch": epoch_dur,
        }, commit=True)
        wandb.run.summary["last_train_loss_epoch"] = epoch_avg
        logger.info(f"epoch {epoch} done: avg_loss={epoch_avg:.4f}, dur={epoch_dur:.1f}s")

        if (epoch + 1) % args.ckpt_every_epoch == 0:
            _save_checkpoint(model, ema, opt, args, epoch, train_steps, checkpoint_dir, logger)
            _preview_sample(
                model, ema, fast_diffusion, args,
                preview_mean, alpha_rank_std, vhat_std,
                device, layout, epoch, logger,
            )

    _save_checkpoint(model, ema, opt, args, args.epochs - 1, train_steps,
                     checkpoint_dir, logger, final=True)
    logger.info("Done.")
    cleanup()


def _save_checkpoint(model, ema, opt, args, epoch, train_steps, checkpoint_dir, logger, final=False):
    base_model = getattr(model, "_orig_mod", model)
    ckpt = {
        "model":       base_model.state_dict(),
        "ema":         ema.state_dict() if args.ema else None,
        "opt":         opt.state_dict(),
        "args":        args,
        "epoch":       epoch,
        "train_steps": train_steps,
    }
    name = "final.pt" if final else f"epoch_{epoch+1:05d}.pt"
    path = os.path.join(checkpoint_dir, name)
    torch.save(ckpt, path)
    logger.info(f"saved checkpoint -> {path}")


@torch.no_grad()
def _preview_sample(model, ema, fast_diffusion, args,
                    preview_mean, alpha_rank_std, vhat_std,
                    device, layout, epoch, logger):
    model_was_training = model.training
    model.eval()
    try:
        B_vis = 4
        H, W = args.input_size
        z = torch.randn(B_vis, args.in_channels, H, W, device=device)
        z = z.contiguous(memory_format=torch.channels_last)
        samples = fast_diffusion.p_sample_loop(
            ema.forward if args.ema else model.forward,
            z.shape, z,
            clip_denoised=False, model_kwargs={}, progress=False, device=device,
        )
        alpha_n   = layout["alpha_n"]
        patch_dim = layout["patch_dim"]
        img_hw    = layout["img_hw"]
        patch     = layout["patch"]

        alpha_samp = samples[:, :, :alpha_n, :]
        V_samp     = samples[:, :, alpha_n:, :]
        alpha_samp = alpha_samp * alpha_rank_std.to(device)[None, :, None, :]
        V_samp     = V_samp * vhat_std.to(device)

        D_vis    = V_samp                                # (B, 3, patch_dim, R) per-sample dictionary
        mean_vis = preview_mean.to(device)               # (3, patch_dim)
        recon = recover_from_alpha_to_image(alpha_samp, D_vis, mean_vis, patch=patch, img_hw=img_hw)
        q05, q95 = recon.min(), recon.max()
        grid = vutils.make_grid(recon, nrow=4, normalize=True, value_range=(q05.item(), q95.item()))
        wandb.log({
            "samples/epoch_preview": wandb.Image(
                grid.permute(1, 2, 0).detach().cpu().numpy(),
                caption=f"epoch {epoch}",
            ),
            "epoch": epoch,
        }, commit=True)
    except Exception as e:
        logger.warning(f"preview at epoch {epoch} failed: {e}")
    finally:
        if model_was_training:
            model.train()


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    _default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.yaml")
    pre.add_argument("--config", type=str,
                     default=_default_cfg if os.path.exists(_default_cfg) else None)
    pre_args, _ = pre.parse_known_args()
    yaml_cfg = _load_yaml(pre_args.config)
    if pre_args.config:
        print(f"[Config] loaded {pre_args.config} ({len(yaml_cfg)} keys)")

    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--hidden-size", type=int, default=1152)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every-epoch", type=int, default=100)
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--learn-sigma", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema", action="store_true", default=False)
    parser.add_argument("--ortho-augment", action="store_true", default=False)
    parser.add_argument("--shard-dir", type=str,
                        default="${DATA_ROOT}/tucker_factors/celeba/no_alignment")
    parser.add_argument("--alpha-stats-path", type=str,
                        default="${DATA_ROOT}/tucker_factors/celeba/no_alignment/alpha_stats_no_alignment_p32_r32.pt")
    parser.add_argument("--vhat-stats-path", type=str,
                        default="${DATA_ROOT}/tucker_factors/celeba/no_alignment/vhat_stats_no_alignment_p32_r32.pt")
    parser.add_argument("--mean-avg-path", type=str,
                        default="${DATA_ROOT}/tucker_factors/celeba/no_alignment/mean_avg_no_alignment_p32_r32.pt")
    parser.add_argument("--results_dir", type=str,
                        default="${DATA_ROOT}/ablation_results/no_alignment")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--input-size", type=int, nargs=2, default=[2048, 32])
    parser.add_argument("--patch-size", type=int, nargs=2, default=[32, 32])
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512, 1024], default=1024)
    parser.add_argument("--sample-steps", type=int, default=250)
    parser.add_argument("--wandb_project", type=str, default="celeba_p32r32_ablation")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")

    if yaml_cfg:
        known = {a.dest for a in parser._actions}
        documented_only = {"compile", "use_bf16", "noise_schedule", "diffusion_steps", "predict_xstart",
                           "patch", "prefetch_factor", "method"}
        filtered, unknown = {}, []
        for k, v in yaml_cfg.items():
            if k in known: filtered[k] = v
            elif k in documented_only: continue
            else: unknown.append(k)
        if unknown: print(f"[Config] ignoring unknown YAML keys: {unknown}")
        if filtered: parser.set_defaults(**filtered)

    args = parser.parse_args()
    main(args)
