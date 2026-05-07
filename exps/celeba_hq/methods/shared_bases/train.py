"""
Train AlphaOnlyDiT on shared_bases (global PCA) projections.

Input data per shard (from all_save_global_pca.py):
  alpha: (B, 3, 1024, 32)
Reconstruction in preview:
  image_patches = alpha @ D[c].T + mean[c]   (D, mean from global_dict.pt)

Checkpoints + W&B sample previews fire on EPOCH boundaries.
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

# Local model (Option A: alpha-only DiT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dit_models_alpha_only import AlphaOnlyDiT

# Re-use diffusion package at celeba_hq root (two levels up from methods/<name>/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from diffusion import create_diffusion

# Re-use ShardedAlphaDataset and a few helpers from our_method/train.py (read-only).
# Load it explicitly by path under a unique module name to avoid the local
# train.py shadowing it.
import importlib.util  # noqa: E402
_OUR_METHOD = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "our_method")
)
_om_spec = importlib.util.spec_from_file_location("our_method_train", os.path.join(_OUR_METHOD, "train.py"))
_om_train = importlib.util.module_from_spec(_om_spec)
sys.modules["our_method_train"] = _om_train
_om_spec.loader.exec_module(_om_train)
ShardedAlphaDataset           = _om_train.ShardedAlphaDataset
requires_grad                 = _om_train.requires_grad
update_ema                    = _om_train.update_ema
recover_from_alpha_to_image   = _om_train.recover_from_alpha_to_image
cleanup                       = _om_train.cleanup
create_logger                 = _om_train.create_logger


def _strip_compile_prefix(sd: dict) -> dict:
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(cfg).__name__}")
    return cfg


def main(args):
    assert torch.cuda.is_available(), "Training requires a GPU."

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index  = len(glob.glob(f"{args.results_dir}/*"))
    experiment_dir    = f"{args.results_dir}/{experiment_index:03d}-AlphaOnlyDiT"
    checkpoint_dir    = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment dir: {experiment_dir}")

    # ---- Load global dict (D, mean) ----
    dct = torch.load(args.dict_path, map_location="cpu", weights_only=False)
    global_D    = dct["D"].float()        # (3, patch_dim, rank)
    global_mean = dct["mean"].float()     # (3, patch_dim)
    logger.info(f"global_dict.pt loaded: D {tuple(global_D.shape)}, mean {tuple(global_mean.shape)}")

    # ---- Load per-(C,R) alpha stats ----
    stats = torch.load(args.alpha_stats_path, map_location="cpu", weights_only=False)
    alpha_rank_std = stats["std"].float()                                   # (3, R)
    logger.info(
        f"alpha stats: std shape {tuple(alpha_rank_std.shape)}, "
        f"range [{alpha_rank_std.min():.4f}, {alpha_rank_std.max():.4f}]"
    )

    # ---- Dataset ----
    dataset = ShardedAlphaDataset(args.shard_dir, preload=False)
    assert len(dataset) >= 100, f"need >=100 images; got {len(dataset)}"
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
    steps_per_epoch = len(loader)
    logger.info(f"dataset={len(dataset)} batches/epoch={steps_per_epoch}")

    device = torch.device("cuda")

    # ---- Model + diffusion ----
    img_hw   = args.image_size
    patch    = args.patch_size[0]
    rank     = args.svd_rank
    model = AlphaOnlyDiT(
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
        learn_sigma=args.learn_sigma,
        timestep_respacing="",
        predict_xstart=True,
    )
    fast_diffusion = create_diffusion(
        learn_sigma=args.learn_sigma,
        timestep_respacing=str(args.sample_steps),
        predict_xstart=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"AlphaOnlyDiT params: {n_params:,} ({n_params/1e6:.2f} M)")
    logger.info(f"args: {args}")

    if args.ema:
        update_ema(ema, model, decay=0)
    model.train(); ema.eval()

    # W&B
    auto_name = (
        f"shared_bases_AlphaOnlyDiT_lr{args.lr}_p{patch}_r{rank}_bs{args.global_batch_size}"
    )
    name = args.wandb_run_name or auto_name
    wandb.init(
        entity=args.wandb_entity or None,
        project=args.wandb_project,
        name=name,
        config={
            "method": "shared_bases",
            "model": "AlphaOnlyDiT",
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

    log_steps   = 0
    running_loss = 0.0
    start_time   = time()

    logger.info(f"Training for {args.epochs} epochs (start_epoch={start_epoch}, train_steps={train_steps}) ...")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"=== epoch {epoch} ===")
        epoch_loss_sum = torch.zeros(1, device=device)
        epoch_batches  = 0
        epoch_t0       = time()

        for raw, _fname in tqdm(loader, desc=f"ep{epoch}"):
            raw = raw.to(device)                                   # (B, 3, alpha_n, R)
            data = raw / alpha_rank_std.to(device)[None, :, None, :]
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

        # ---- end of epoch ----
        epoch_avg = (epoch_loss_sum / max(1, epoch_batches)).item()
        epoch_dur = time() - epoch_t0
        wandb.log({
            "epoch":             epoch,
            "train/loss_epoch":  epoch_avg,
            "train/time_epoch":  epoch_dur,
        }, commit=True)
        wandb.run.summary["last_train_loss_epoch"] = epoch_avg
        logger.info(f"epoch {epoch} done: avg_loss={epoch_avg:.4f}, dur={epoch_dur:.1f}s")

        # ---- ckpt + preview every N epochs ----
        if (epoch + 1) % args.ckpt_every_epoch == 0:
            _save_checkpoint(model, ema, opt, args, epoch, train_steps, checkpoint_dir, logger)
            _preview_sample(
                model, ema, fast_diffusion, args,
                global_D, global_mean, alpha_rank_std,
                device, img_hw, patch, rank, epoch, logger,
            )

    # final
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
                    global_D, global_mean, alpha_rank_std,
                    device, img_hw, patch, rank, epoch, logger):
    model_was_training = model.training
    model.eval()
    try:
        B_vis = 4
        alpha_n = (img_hw // patch) ** 2
        z = torch.randn(B_vis, 3, alpha_n, rank, device=device)
        z = z.contiguous(memory_format=torch.channels_last)
        samples = fast_diffusion.p_sample_loop(
            ema.forward if args.ema else model.forward,
            z.shape, z,
            clip_denoised=False, model_kwargs={}, progress=False, device=device,
        )
        # de-normalize
        samples = samples * alpha_rank_std.to(device)[None, :, None, :]
        # reconstruct: alpha @ D.T + mean
        D    = global_D.to(device)        # (3, d, R)
        mean = global_mean.to(device)     # (3, d)
        recon = recover_from_alpha_to_image(samples, D, mean, patch=patch, img_hw=img_hw)
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
    parser.add_argument("--shard-dir", type=str,
                        default="/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/celeba/shared_bases")
    parser.add_argument("--alpha-stats-path", type=str,
                        default="/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/celeba/shared_bases/alpha_stats_global_pca_p32_r32.pt")
    parser.add_argument("--dict-path", type=str,
                        default="/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/celeba/shared_bases/global_dict.pt")
    parser.add_argument("--results_dir", type=str,
                        default="/anvil/projects/x-eng260004/factor_diffusion/ablation_results/shared_bases")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--patch-size", type=int, nargs=2, default=[32, 32])
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
