from __future__ import annotations

import argparse
import math
import os
from glob import glob
from pathlib import Path
from typing import Optional

import accelerate
import numpy as np
import torch
import torch.nn as nn
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
    return torch.stack([gx, gy], dim=0)  # (2, H, W)


def make_cylinder_mask(cx: int, cy: int, r: int, h: int, w: int) -> torch.Tensor:
    ys = torch.arange(h).view(h, 1).expand(h, w)
    xs = torch.arange(w).view(1, w).expand(h, w)
    return ((xs - cx) ** 2 + (ys - cy) ** 2 <= r * r).float()


def _mmap_load_all(shard_paths: list[str], desc: str) -> list[list[dict]]:
    shards = []
    for sp in tqdm(shard_paths, desc=desc, leave=False):
        shards.append(torch.load(sp, map_location="cpu", mmap=True, weights_only=False))
    return shards


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
    def __init__(self, *, t_in: int, modes: int, hidden_channels: int, n_layers: int):
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


def per_sample_metrics(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff = (pred - target).reshape(pred.size(0), -1)
    norm = target.reshape(target.size(0), -1)
    rel_l1 = (diff.abs().sum(-1) + eps) / (norm.abs().sum(-1) + eps)
    rel_l2 = (diff.pow(2).sum(-1) + eps).sqrt() / (norm.pow(2).sum(-1) + eps).sqrt()
    r_mse = diff.pow(2).mean(-1).sqrt()
    return rel_l1, rel_l2, r_mse


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
        per_step_l1 = (diff_step.abs().sum(-1) + eps) / (norm_step.abs().sum(-1) + eps)
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
            {"preds": preds_all, "truths": truths_all, "init_mode": init_mode,
             "t_in": t_in, "t_rollout": t_rollout, "n_clips": preds_all.size(0),
             "shape": tuple(preds_all.shape), "dtype": "float16"},
            save_path,
        )
        out["samples_saved"] = save_path
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval FNO-2D on the 2D Karman vortex test set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-dir", type=str,
                        default="${DATA_ROOT}/bkx8728/karman_vortex_2d/test_data")
    parser.add_argument("--mixed-precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--t-in", type=int, default=10)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=193)
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--t-rollout", type=int, default=30)
    parser.add_argument("--eval-max-clips", type=int, default=0,
                        help="cap test clips (0 = use all)")
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--eval-init-mode", type=str, default="history",
                        choices=["history", "single"])
    parser.add_argument("--save-samples-dir", type=str, default="")
    args = parser.parse_args()
    if args.eval_max_clips == 0:
        args.eval_max_clips = None
    return args


def main():
    args = parse_args()
    accelerator = accelerate.Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    test_shards = list_shards(args.test_dir, "test_shard_")
    test_set = KarmanRolloutTestset(test_shards, t_in=args.t_in)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=0)

    model = FNOKarman(t_in=args.t_in, modes=args.modes,
                      hidden_channels=args.hidden_channels, n_layers=args.n_layers)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ckpt["model"].items() if isinstance(v, torch.Tensor)}
    if args.use_ema and ckpt.get("ema"):
        sd = {k: ckpt["ema"].get(k, v) for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device)

    metrics = evaluate_rollout(
        model, test_loader, device=device,
        t_in=args.t_in, t_rollout=args.t_rollout,
        max_clips=args.eval_max_clips,
        init_mode=args.eval_init_mode,
        save_samples_dir=args.save_samples_dir or None,
    )

    if accelerator.is_main_process:
        def fmt(key: str) -> str:
            m = metrics[f"test_{key}"]
            sem = metrics[f"test_{key}_sem"]
            return f"{m:.3f} ± {sem:.1e}" if abs(m) >= 1.0 else f"{m:.4f} ± {sem:.1e}"
        print(f"clips={metrics['test_clips']}")
        print("                  Avg Rel L1               Avg Rel L2               Avg rMSE")
        print(f"one_step          {fmt('one_step_relL1'):<25}{fmt('one_step_relL2'):<25}{fmt('one_step_rMSE')}")
        print(f"rollout({args.t_rollout}) [sum] {fmt('rollout_relL1'):<25}{fmt('rollout_relL2'):<25}{fmt('rollout_rMSE')}")
        per_step = metrics.get("test_per_step_relL2")
        if per_step:
            milestones = [1, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, len(per_step)]
            print("per-step relL2 (1-indexed):")
            for k in milestones:
                if 1 <= k <= len(per_step):
                    print(f"  step={k:3d}  relL2={per_step[k - 1]:.4f}")


if __name__ == "__main__":
    main()
