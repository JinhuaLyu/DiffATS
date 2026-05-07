from __future__ import annotations

import argparse
import math
import os
import random
import time
from glob import glob
from pathlib import Path
from typing import Optional

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from neuralop.models import FNO
except ImportError as exc:
    raise ImportError(
        "This script requires neuraloperator. Install it with `pip install neuraloperator`."
    ) from exc


T_TOTAL = 201
H = 128
W = 128


def list_shards(root: str, prefix: str) -> list[str]:
    paths = sorted(glob(os.path.join(root, f"{prefix}*.pt")))
    if not paths:
        raise FileNotFoundError(f"no shards matching '{prefix}*.pt' under {root}")
    return paths


def make_grid_channels(h: int, w: int) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, h)
    xs = torch.linspace(-1.0, 1.0, w)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=0)


def make_cylinder_mask(cx: int, cy: int, r: int, h: int, w: int) -> torch.Tensor:
    ys = torch.arange(h).view(h, 1).expand(h, w)
    xs = torch.arange(w).view(1, w).expand(h, w)
    return ((xs - cx) ** 2 + (ys - cy) ** 2 <= r * r).float()


def _mmap_load_all(shard_paths: list[str], desc: str) -> list[list[dict]]:
    shards = []
    for sp in tqdm(shard_paths, desc=desc, leave=False):
        shards.append(torch.load(sp, map_location="cpu", mmap=True, weights_only=False))
    return shards


class KarmanMmapTrainset(Dataset):
    def __init__(
        self,
        shard_paths: list[str],
        t_in: int,
        rollout_steps: int,
        samples_per_clip: int = 1,
    ):
        super().__init__()
        if not shard_paths:
            raise FileNotFoundError("no train shards")
        self.t_in = int(t_in)
        self.rollout_steps = int(rollout_steps)
        self.samples_per_clip = int(samples_per_clip)
        self._shards = _mmap_load_all(shard_paths, desc="mmap train shards")
        self._index: list[tuple[int, int]] = []
        self._meta: list[tuple[int, int, int]] = []
        for s_idx, shard in enumerate(self._shards):
            for c_idx, sample in enumerate(shard):
                T = sample["vor"].shape[0]
                if T < self.t_in + self.rollout_steps:
                    continue
                self._index.append((s_idx, c_idx))
                self._meta.append((int(sample["cx"]), int(sample["cy"]), int(sample["r"])))

    def __len__(self) -> int:
        return len(self._index) * self.samples_per_clip

    def __getitem__(self, idx: int):
        clip_idx = idx % len(self._index)
        s_idx, c_idx = self._index[clip_idx]
        clip = self._shards[s_idx][c_idx]["vor"]
        T = clip.shape[0]
        K = self.rollout_steps
        t = int(torch.randint(self.t_in, T - K + 1, (1,)).item())
        past = clip[t - self.t_in : t].clone()
        future = clip[t : t + K].clone()
        cx, cy, r = self._meta[clip_idx]
        mask = make_cylinder_mask(cx, cy, r, H, W).unsqueeze(0)
        return past, future, mask


class KarmanRolloutTestset(Dataset):
    def __init__(self, shard_paths: list[str], t_in: int):
        if not shard_paths:
            raise FileNotFoundError("no test shards")
        self.t_in = int(t_in)
        self._shards = _mmap_load_all(shard_paths, desc="mmap test shards")
        self._index: list[tuple[int, int]] = []
        self._meta: list[tuple[int, int, int]] = []
        for s_idx, shard in enumerate(self._shards):
            for c_idx, sample in enumerate(shard):
                self._index.append((s_idx, c_idx))
                self._meta.append((int(sample["cx"]), int(sample["cy"]), int(sample["r"])))
        self._grid = make_grid_channels(H, W)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        s_idx, c_idx = self._index[idx]
        clip = self._shards[s_idx][c_idx]["vor"].clone()
        cx, cy, r = self._meta[idx]
        mask = make_cylinder_mask(cx, cy, r, H, W)
        return clip, self._grid, mask


class FNOKarman(nn.Module):
    def __init__(
        self,
        *,
        t_in: int,
        modes: int,
        hidden_channels: int,
        n_layers: int,
    ):
        super().__init__()
        self.t_in = int(t_in)
        self.fno = FNO(
            n_modes=(modes, modes),
            hidden_channels=hidden_channels,
            in_channels=t_in + 3,
            out_channels=1,
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = self.fno(x.float())
        return out.to(in_dtype)


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff = (pred - target).reshape(pred.size(0), -1)
    norm = target.reshape(target.size(0), -1)
    return ((diff.pow(2).sum(-1) + eps).sqrt() / (norm.pow(2).sum(-1) + eps).sqrt()).mean()


def per_sample_metrics(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff = (pred - target).reshape(pred.size(0), -1)
    norm = target.reshape(target.size(0), -1)
    rel_l1 = (diff.abs().sum(-1) + eps) / (norm.abs().sum(-1) + eps)
    rel_l2 = (diff.pow(2).sum(-1) + eps).sqrt() / (norm.pow(2).sum(-1) + eps).sqrt()
    r_mse = diff.pow(2).mean(-1).sqrt()
    return rel_l1, rel_l2, r_mse


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        state = model.state_dict()
        for k, shadow_v in self.shadow.items():
            v = state[k].detach()
            if shadow_v.device != v.device:
                shadow_v = shadow_v.to(v.device)
                self.shadow[k] = shadow_v
            shadow_v.mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module):
        state = model.state_dict()
        merged = {}
        for k, v in state.items():
            if not isinstance(v, torch.Tensor):
                continue
            sv = self.shadow.get(k)
            if sv is None:
                merged[k] = v
            elif sv.device != v.device:
                merged[k] = sv.to(v.device)
            else:
                merged[k] = sv
        model.load_state_dict(merged, strict=False)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def maybe_cuda_peak_memory_gb(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        "peak_mem_alloc_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "peak_mem_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
    }


@torch.no_grad()
def evaluate_rollout(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    t_in: int,
    t_rollout: int,
    max_clips: Optional[int] = None,
    init_mode: str = "history",
    save_samples_dir: Optional[str] = None,
) -> dict[str, float]:
    model.eval()
    metric_keys = [
        "one_step_relL1", "one_step_relL2", "one_step_rMSE",
        "rollout_relL1", "rollout_relL2", "rollout_rMSE",
    ]
    sums = {k: 0.0 for k in metric_keys}
    sumsq = {k: 0.0 for k in metric_keys}
    per_step_relL2_sum: Optional[torch.Tensor] = None
    seen = 0
    save_preds: list[torch.Tensor] = []
    save_truths: list[torch.Tensor] = []
    for clips, grids, masks in test_loader:
        clips = clips.to(device, non_blocking=True)
        grids = grids.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        B, T, _, _ = clips.shape
        if init_mode == "single":
            steps = min(t_rollout, T - 1)
            past = clips[:, 0:1].expand(B, t_in, *clips.shape[2:]).contiguous()
            truth_start = 1
        else:
            steps = min(t_rollout, T - t_in)
            past = clips[:, :t_in]
            truth_start = t_in
        preds = []
        for k in range(steps):
            inp = torch.cat([past, grids, masks.unsqueeze(1)], dim=1)
            nxt = model(inp)[:, 0]
            preds.append(nxt)
            past = torch.cat([past[:, 1:], nxt.unsqueeze(1)], dim=1)
        preds = torch.stack(preds, dim=1)
        truth = clips[:, truth_start : truth_start + steps]

        l1_os, l2_os, rmse_os = per_sample_metrics(preds[:, :1], truth[:, :1])

        diff_step = (preds - truth).reshape(B, steps, -1)
        norm_step = truth.reshape(B, steps, -1)
        eps = 1e-8
        per_step_l1 = ((diff_step.abs().sum(-1) + eps)
                       / (norm_step.abs().sum(-1) + eps))
        per_step_l2 = ((diff_step.pow(2).sum(-1) + eps).sqrt()
                       / (norm_step.pow(2).sum(-1) + eps).sqrt())
        per_step_rmse = diff_step.pow(2).mean(-1).sqrt()

        l1_ro = per_step_l1.sum(dim=-1)
        l2_ro = per_step_l2.sum(dim=-1)
        rmse_ro = per_step_rmse.sum(dim=-1)

        per_metric = {
            "one_step_relL1": l1_os, "one_step_relL2": l2_os, "one_step_rMSE": rmse_os,
            "rollout_relL1": l1_ro, "rollout_relL2": l2_ro, "rollout_rMSE": rmse_ro,
        }
        for k, v in per_metric.items():
            sums[k] += v.sum().item()
            sumsq[k] += v.pow(2).sum().item()

        if per_step_relL2_sum is None:
            per_step_relL2_sum = per_step_l2.sum(dim=0)
        else:
            per_step_relL2_sum += per_step_l2.sum(dim=0)

        if save_samples_dir is not None:
            save_preds.append(preds.detach().to(dtype=torch.float16, device="cpu"))
            save_truths.append(truth.detach().to(dtype=torch.float16, device="cpu"))

        seen += B
        if max_clips is not None and seen >= max_clips:
            break

    denom = max(seen, 1)
    out: dict[str, float] = {}
    for k in metric_keys:
        mean = sums[k] / denom
        var = max(sumsq[k] / denom - mean * mean, 0.0)
        std = math.sqrt(var)
        sem = std / math.sqrt(max(denom, 1))
        out[f"test_{k}"] = mean
        out[f"test_{k}_std"] = std
        out[f"test_{k}_sem"] = sem
    out["test_clips"] = seen
    if per_step_relL2_sum is not None:
        out["test_per_step_relL2"] = (per_step_relL2_sum / denom).cpu().tolist()
    if save_samples_dir is not None and save_preds:
        os.makedirs(save_samples_dir, exist_ok=True)
        preds_all = torch.cat(save_preds, dim=0)
        truths_all = torch.cat(save_truths, dim=0)
        save_path = os.path.join(save_samples_dir, "samples.pt")
        torch.save(
            {
                "preds": preds_all,
                "truths": truths_all,
                "init_mode": init_mode,
                "t_in": t_in,
                "t_rollout": t_rollout,
                "n_clips": preds_all.size(0),
                "shape": tuple(preds_all.shape),
                "dtype": "float16",
            },
            save_path,
        )
        out["samples_saved"] = save_path
    return out


def save_checkpoint(
    path: str,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.shadow,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ckpt["model"].items() if isinstance(v, torch.Tensor)}
    model.load_state_dict(sd, strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def train(args: argparse.Namespace):
    accelerator = accelerate.Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    train_shards = list_shards(args.data_dir, "shard_")
    test_shards = list_shards(args.test_dir, "test_shard_")

    train_set = KarmanMmapTrainset(
        shard_paths=train_shards,
        t_in=args.t_in,
        rollout_steps=args.rollout_steps,
        samples_per_clip=args.samples_per_clip,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    test_set = KarmanRolloutTestset(test_shards, t_in=args.t_in)
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = FNOKarman(
        t_in=args.t_in,
        modes=args.modes,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
    )
    n_params = count_parameters(model)
    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(len(train_set) // (args.batch_size * accelerator.num_processes), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * args.epochs, eta_min=args.lr * 0.01
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    grid = make_grid_channels(H, W).to(device)

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if accelerator.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"device={device} mixed_precision={accelerator.mixed_precision} "
            f"params={n_params} ({n_params/1e6:.2f}M) "
            f"rollout_steps={args.rollout_steps} t_in={args.t_in} "
            f"train_clips_per_epoch~{steps_per_epoch * args.batch_size * accelerator.num_processes}"
        )

    step = 0
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_l2 = 0.0
        epoch_batches = 0
        progress = tqdm(
            train_loader,
            disable=not accelerator.is_main_process,
            desc=f"epoch {epoch}/{args.epochs}",
            leave=True,
            total=steps_per_epoch,
        )
        t0 = time.time()
        K = args.rollout_steps
        for past, future, mask in progress:
            B = past.size(0)
            grid_b = grid.unsqueeze(0).expand(B, -1, -1, -1)

            optimizer.zero_grad(set_to_none=True)
            cur_past = past
            preds = []
            for _ in range(K):
                inp = torch.cat([cur_past, grid_b, mask], dim=1)
                nxt = model(inp)
                preds.append(nxt)
                cur_past = torch.cat([cur_past[:, 1:], nxt], dim=1)
            pred_seq = torch.cat(preds, dim=1)

            mse = F.mse_loss(pred_seq, future)
            l2 = relative_l2(pred_seq, future)
            loss = l2 if args.loss == "rel_l2" else mse
            accelerator.backward(loss)
            if args.grad_clip is not None and args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(accelerator.unwrap_model(model))

            epoch_loss += float(mse.detach().item())
            epoch_l2 += float(l2.detach().item())
            epoch_batches += 1
            step += 1
            if accelerator.is_main_process:
                progress.set_postfix(
                    mse=f"{mse.detach().item():.6f}",
                    l2=f"{l2.detach().item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    step=step,
                )

        avg_mse = epoch_loss / max(epoch_batches, 1)
        avg_l2 = epoch_l2 / max(epoch_batches, 1)
        if accelerator.is_main_process:
            mem = maybe_cuda_peak_memory_gb(device)
            mem_str = " ".join(f"{k}={v:.3f}" for k, v in mem.items())
            print(
                f"epoch={epoch} step={step} mse={avg_mse:.6f} relL2={avg_l2:.4f} "
                f"epoch_time={time.time() - t0:.1f}s {mem_str}".strip()
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        unwrapped = accelerator.unwrap_model(model)
        if accelerator.is_main_process and (epoch % args.save_every == 0 or epoch == args.epochs):
            save_checkpoint(str(ckpt_dir / f"epoch_{epoch}.pt"), unwrapped, ema, optimizer, epoch, args)
        if accelerator.is_main_process and avg_l2 < best_loss:
            best_loss = avg_l2
            save_checkpoint(str(ckpt_dir / "best.pt"), unwrapped, ema, optimizer, epoch, args)

        if accelerator.is_main_process and (epoch % args.eval_every == 0 or epoch == args.epochs):
            ema_model = FNOKarman(
                t_in=args.t_in,
                modes=args.modes,
                hidden_channels=args.hidden_channels,
                n_layers=args.n_layers,
            ).to(device)
            ema.copy_to(ema_model)
            metrics = evaluate_rollout(
                ema_model,
                test_loader,
                device=device,
                t_in=args.t_in,
                t_rollout=args.t_rollout,
                max_clips=args.eval_max_clips,
            )
            print(
                f"[eval] epoch={epoch} clips={metrics['test_clips']}\n"
                f"  one_step  relL1={metrics['test_one_step_relL1']:.4f} "
                f"relL2={metrics['test_one_step_relL2']:.4f} "
                f"rMSE={metrics['test_one_step_rMSE']:.6f}\n"
                f"  rollout({args.t_rollout})  relL1={metrics['test_rollout_relL1']:.4f} "
                f"relL2={metrics['test_rollout_relL2']:.4f} "
                f"rMSE={metrics['test_rollout_rMSE']:.6f}"
            )
            del ema_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FNO-2D for the 2D Karman vortex dataset")
    parser.add_argument("--data-dir", type=str, default="${DATA_ROOT}/bkx8728/karman_vortex_2d")
    parser.add_argument("--test-dir", type=str, default="${DATA_ROOT}/bkx8728/karman_vortex_2d/test_data")
    parser.add_argument("--run-dir", type=str, default="karman_fno_runs")
    parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--t-in", type=int, default=10, help="number of past frames stacked as input channels")
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=193)
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-clip", type=int, default=1,
                        help="number of random (clip, t) samples per clip per epoch "
                             "(virtual replication of the index; t is sampled in __getitem__)")
    parser.add_argument("--rollout-steps", type=int, default=4,
                        help="autoregressive rollout horizon during TRAINING; the model "
                             "predicts K future frames with backprop through every step.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--loss", type=str, default="rel_l2", choices=["rel_l2", "mse"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--t-rollout", type=int, default=30, help="autoregressive rollout horizon at eval time")
    parser.add_argument("--eval-max-clips", type=int, default=200,
                        help="cap eval clips per check (set 0 to use all)")

    args = parser.parse_args()
    if args.eval_max_clips == 0:
        args.eval_max_clips = None
    return args


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
