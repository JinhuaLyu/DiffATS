from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from train_burgers_fno import (
    BurgersRolloutTestset,
    FNOBurgers,
    H,
    W,
    list_shards,
    make_grid_channels,
)


def per_sample_metrics(pred: torch.Tensor, truth: torch.Tensor, eps: float = 1e-12):
    """pred, truth: (B, T, H, W). Returns three (B,) tensors: rL1, rL2, rMSE."""
    diff = (pred - truth).reshape(pred.size(0), -1)
    tgt = truth.reshape(truth.size(0), -1)

    num_l1 = diff.abs().sum(-1)
    den_l1 = tgt.abs().sum(-1).clamp_min(eps)
    rL1 = num_l1 / den_l1

    num_l2 = diff.pow(2).sum(-1).sqrt()
    den_l2 = tgt.pow(2).sum(-1).sqrt().clamp_min(eps)
    rL2 = num_l2 / den_l2

    rMSE = diff.pow(2).mean(-1).sqrt()

    return rL1, rL2, rMSE


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    t_in: int,
    t_rollout: int,
    seed_mode: str = "lookback",
):
    model.eval()
    rL1_all, rL2_all, rMSE_all = [], [], []
    seen = 0
    for clips, grids in tqdm(test_loader, desc="eval", leave=False):
        clips = clips.to(device, non_blocking=True)
        grids = grids.to(device, non_blocking=True)

        B, T, _, _ = clips.shape
        if seed_mode == "lookback":
            # Use t_in real past frames, then predict t_rollout future frames.
            steps = min(t_rollout, T - t_in)
            past = clips[:, :t_in]
            target_start = t_in
        elif seed_mode == "single":
            # Only frame 0 is observed; repeat it t_in times to fill the model's
            # past-frame buffer, then autoregressively predict frames 1..steps.
            steps = min(t_rollout, T - 1)
            past = clips[:, 0:1].expand(-1, t_in, -1, -1).contiguous()
            target_start = 1
        else:
            raise ValueError(f"unknown seed_mode={seed_mode!r}")
        preds = []
        for _ in range(steps):
            inp = torch.cat([past, grids], dim=1)
            nxt = model(inp)[:, 0]
            preds.append(nxt)
            past = torch.cat([past[:, 1:], nxt.unsqueeze(1)], dim=1)
        preds = torch.stack(preds, dim=1)
        truth = clips[:, target_start : target_start + steps]

        rL1, rL2, rMSE = per_sample_metrics(preds, truth)
        rL1_all.append(rL1.cpu())
        rL2_all.append(rL2.cpu())
        rMSE_all.append(rMSE.cpu())
        seen += B

    rL1_all = torch.cat(rL1_all)
    rL2_all = torch.cat(rL2_all)
    rMSE_all = torch.cat(rMSE_all)
    return {
        "n": int(rL1_all.numel()),
        "rL1_mean": float(rL1_all.mean()),
        "rL1_std": float(rL1_all.std(unbiased=False)),
        "rL2_mean": float(rL2_all.mean()),
        "rL2_std": float(rL2_all.std(unbiased=False)),
        "rMSE_mean": float(rMSE_all.mean()),
        "rMSE_std": float(rMSE_all.std(unbiased=False)),
    }


def fmt(mean: float, std: float) -> str:
    return f"{mean:.5f} ± {std:.2e}"


def load_model_with_optional_ema(ckpt_path: str, model: nn.Module, use_ema: bool):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = dict(ckpt["model"])
    raw.pop("_metadata", None)
    model.load_state_dict(raw, strict=True)
    if use_ema:
        ema_shadow = dict(ckpt.get("ema") or {})
        ema_shadow.pop("_metadata", None)
        if not ema_shadow:
            raise RuntimeError(f"--use-ema set but no EMA shadow in {ckpt_path}")
        state = model.state_dict()
        merged = {k: ema_shadow.get(k, state[k]) for k in state.keys() if k != "_metadata"}
        model.load_state_dict(merged, strict=True)
    return ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval FNO burgers checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-dir", type=str, default="/projects/p32954/jinhua_data/burgers_2d_test")
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Load EMA weights (default; pass --no-use-ema to use raw weights)")
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--t-in", type=int, default=10)
    parser.add_argument("--t-rollout", type=int, default=30)
    parser.add_argument("--seed-mode", type=str, default="lookback", choices=["lookback", "single"],
                        help="lookback: feed t_in real past frames (default). "
                             "single: feed only frame 0 (repeated t_in times) and roll out 1..t_rollout.")
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=190)
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    # Allow the checkpoint's args to override architecture / t_in if present.
    ckpt_peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ck_args = ckpt_peek.get("args", {}) or {}
    t_in = int(ck_args.get("t_in", args.t_in))
    modes = int(ck_args.get("modes", args.modes))
    hidden_channels = int(ck_args.get("hidden_channels", args.hidden_channels))
    n_layers = int(ck_args.get("n_layers", args.n_layers))
    print(f"[eval] t_in={t_in} modes={modes} hidden={hidden_channels} n_layers={n_layers} t_rollout={args.t_rollout}")
    del ckpt_peek

    test_shards = list_shards(args.test_dir, "test_shard_")
    test_set = BurgersRolloutTestset(test_shards, t_in=t_in)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"[eval] test clips: {len(test_set)} (= shards x 100 x 2 components)")

    model = FNOBurgers(t_in=t_in, modes=modes, hidden_channels=hidden_channels, n_layers=n_layers)
    load_model_with_optional_ema(args.checkpoint, model, use_ema=args.use_ema)
    model = model.to(device)
    print(f"[eval] loaded checkpoint: {args.checkpoint}  use_ema={args.use_ema}")

    metrics = evaluate(model, test_loader, device, t_in=t_in, t_rollout=args.t_rollout,
                       seed_mode=args.seed_mode)

    print()
    print("=" * 62)
    print(f"  Test clips evaluated:  {metrics['n']}")
    print(f"  Seed mode:             {args.seed_mode}")
    print(f"  Rollout horizon:       {args.t_rollout}")
    print("-" * 62)
    print(f"  Average Relative L1:   {fmt(metrics['rL1_mean'],  metrics['rL1_std'])}")
    print(f"  Average Relative L2:   {fmt(metrics['rL2_mean'],  metrics['rL2_std'])}")
    print(f"  Average rMSE:          {fmt(metrics['rMSE_mean'], metrics['rMSE_std'])}")
    print("=" * 62)

    if args.out_json:
        import json
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "use_ema": args.use_ema,
                "t_rollout": args.t_rollout,
                **metrics,
            }, f, indent=2)
        print(f"[eval] wrote {args.out_json}")


if __name__ == "__main__":
    main()
