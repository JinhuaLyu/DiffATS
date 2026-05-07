from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from neuralop.losses import H1Loss, LpLoss
from neuralop.models import FNO


def patch_spectral_conv_for_bf16():
    from neuralop.layers.spectral_convolution import SpectralConv
    _orig_forward = SpectralConv.forward

    def _patched(self, x, *args, **kwargs):
        in_dtype = x.dtype
        if in_dtype == torch.float32:
            return _orig_forward(self, x, *args, **kwargs)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            out = _orig_forward(self, x.float(), *args, **kwargs)
        return out.to(in_dtype)

    SpectralConv.forward = _patched


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", default="/scratch/bkx8728/reaction_1d/reaction_1d_train.pt")
    p.add_argument("--test_path", default="/scratch/bkx8728/reaction_1d/reaction_1d_test.pt")
    p.add_argument("--n_modes", type=int, default=128)
    p.add_argument("--hidden_channels", type=int, default=696)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--use_amp", action="store_true",
                   help="enable bf16 mixed-precision autocast on forward")
    p.add_argument("--subsample_t", type=int, default=0,
                   help="K random pairs per traj per epoch; -1 = 1 random pair per traj; 0 = enumerate all (T-1) pairs per traj (full)")
    p.add_argument("--compile", action="store_true",
                   help="wrap model with torch.compile for kernel fusion")
    p.add_argument("--eval_every", type=int, default=5,
                   help="run rollout eval every N epochs")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--time_budget_seconds", type=float, default=20000.0)
    p.add_argument("--rollout_eval_batch", type=int, default=128,
                   help="batch size for full-rollout evaluation on test set")
    p.add_argument("--ckp_dir", default="/gpfs/home/bkx8728/Tensor_factor/1dscripts/fno_reaction/ckp")
    p.add_argument("--run_name", default="fno_reaction_1d_ar")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def load_full(path: str):
    print(f"[data] loading {path} ...", flush=True)
    d = torch.load(path, map_location="cpu", weights_only=False)
    traj = d["tensor"].clone()
    nu = d["nu"].float().clone()
    rho = d["rho"].float().clone()
    log_nu = torch.log10(nu.clamp_min(1e-12))
    log_rho = torch.log10(rho.clamp_min(1e-12))
    print(f"[data] traj {tuple(traj.shape)}  log_nu {tuple(log_nu.shape)}  log_rho {tuple(log_rho.shape)}", flush=True)
    return traj, log_nu, log_rho


class Stats:
    def __init__(self, x: torch.Tensor):
        self.mean = float(x.mean())
        self.std = float(x.std()) + 1e-8

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}


def train_one_epoch(model, traj_n, lnu_n, lrho_n, optimizer, h1_loss,
                    batch_size, device, epoch, time_start, time_budget,
                    use_amp=False, subsample_t=0):
    N, T, X = traj_n.shape
    if subsample_t == -1:
        total_samples = N
        n_idx_all = torch.randperm(N, device=device)
        t_idx_all = torch.randint(0, T - 1, (N,), device=device)
    elif subsample_t and subsample_t > 0:
        total_samples = N * subsample_t
        n_idx_all = torch.randint(0, N, (total_samples,), device=device)
        t_idx_all = torch.randint(0, T - 1, (total_samples,), device=device)
    else:
        total_samples = N * (T - 1)
        indices = torch.randperm(total_samples, device=device)
        n_idx_all = indices // (T - 1)
        t_idx_all = indices % (T - 1)

    iters = total_samples // batch_size
    model.train()
    running_loss = 0.0
    n_seen = 0
    log_every = max(1, iters // 10)

    for step in range(iters):
        s = step * batch_size
        e = s + batch_size
        n_idx = n_idx_all[s:e]
        t_idx = t_idx_all[s:e]

        u_in = traj_n[n_idx, t_idx]
        u_out = traj_n[n_idx, t_idx + 1]
        nu_ch = lnu_n[n_idx].unsqueeze(-1).expand(-1, X)
        rho_ch = lrho_n[n_idx].unsqueeze(-1).expand(-1, X)

        x = torch.stack([u_in, nu_ch, rho_ch], dim=1)
        y = u_out.unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                delta = model(x)
                pred = u_in.unsqueeze(1) + delta
                loss = h1_loss(pred, y)
        else:
            delta = model(x)
            pred = u_in.unsqueeze(1) + delta
            loss = h1_loss(pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        n_seen += batch_size

        if (step + 1) % log_every == 0:
            elapsed = time.time() - time_start
            print(f"  [ep {epoch} step {step+1}/{iters}] "
                  f"loss={running_loss/n_seen:.4e}  t={elapsed:.0f}s", flush=True)
            if elapsed > time_budget:
                print("  [stop] time budget reached mid-epoch", flush=True)
                return running_loss / n_seen, True

    return running_loss / n_seen, False


@torch.no_grad()
def rollout_eval(model, traj_n, lnu_n, lrho_n, h1_loss, l2_loss, batch_size, device, use_amp=False):
    model.eval()
    N, T, X = traj_n.shape

    h1_per_step = torch.zeros(T - 1, device=device)
    l2_per_step = torch.zeros(T - 1, device=device)
    full_l2 = 0.0
    n_total = 0

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        u_curr = traj_n[batch_start:batch_end, 0, :]
        nu_ch = lnu_n[batch_start:batch_end].unsqueeze(-1).expand(-1, X)
        rho_ch = lrho_n[batch_start:batch_end].unsqueeze(-1).expand(-1, X)
        gt = traj_n[batch_start:batch_end]

        traj_pred = [u_curr]
        for t in range(T - 1):
            x = torch.stack([u_curr, nu_ch, rho_ch], dim=1)
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    delta = model(x).squeeze(1)
            else:
                delta = model(x).squeeze(1)
            u_curr = u_curr + delta.float()
            traj_pred.append(u_curr)

            target_t = gt[:, t + 1, :].unsqueeze(1)
            pred_t = u_curr.unsqueeze(1)
            h1_per_step[t] += h1_loss(pred_t, target_t).item()
            l2_per_step[t] += l2_loss(pred_t, target_t).item()

        traj_pred_t = torch.stack(traj_pred, dim=1)
        b = traj_pred_t.size(0)
        denom = gt.flatten(1).norm(dim=-1).clamp_min(1e-8)
        num = (traj_pred_t - gt).flatten(1).norm(dim=-1)
        full_l2 += (num / denom).sum().item()
        n_total += b

    avg_h1_per_step = (h1_per_step / N).cpu()
    avg_l2_per_step = (l2_per_step / N).cpu()
    full_l2_mean = full_l2 / n_total
    return avg_h1_per_step, avg_l2_per_step, full_l2_mean


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    Path(args.ckp_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)

    traj_tr, lnu_tr, lrho_tr = load_full(args.train_path)
    traj_te, lnu_te, lrho_te = load_full(args.test_path)

    u_stats = Stats(traj_tr)
    nu_stats = Stats(lnu_tr)
    rho_stats = Stats(lrho_tr)
    print(f"[stats] u       mean={u_stats.mean:.4f} std={u_stats.std:.4f}", flush=True)
    print(f"[stats] log_nu  mean={nu_stats.mean:.4f} std={nu_stats.std:.4f}", flush=True)
    print(f"[stats] log_rho mean={rho_stats.mean:.4f} std={rho_stats.std:.4f}", flush=True)

    print("[setup] uploading data to GPU ...", flush=True)
    traj_tr_n = u_stats.encode(traj_tr).to(device, non_blocking=True)
    lnu_tr_n = nu_stats.encode(lnu_tr).to(device, non_blocking=True)
    lrho_tr_n = rho_stats.encode(lrho_tr).to(device, non_blocking=True)
    traj_te_n = u_stats.encode(traj_te).to(device, non_blocking=True)
    lnu_te_n = nu_stats.encode(lnu_te).to(device, non_blocking=True)
    lrho_te_n = rho_stats.encode(lrho_te).to(device, non_blocking=True)
    del traj_tr, traj_te
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(f"[setup] GPU mem after data upload: free={free/1e9:.1f}GB / total={total/1e9:.1f}GB", flush=True)

    N_tr, T, X = traj_tr_n.shape
    print(f"[data] train: {N_tr} trajs x {T} steps x {X} pts  ({N_tr*(T-1)} pair samples)", flush=True)
    print(f"[data] test:  {traj_te_n.shape[0]} trajs", flush=True)

    model = FNO(
        n_modes=(args.n_modes,),
        in_channels=3,
        out_channels=1,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        positional_embedding="grid",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] FNO1d  modes={args.n_modes}  hidden={args.hidden_channels}  layers={args.n_layers}  amp={args.use_amp}  params={n_params/1e6:.2f}M", flush=True)

    if args.use_amp:
        patch_spectral_conv_for_bf16()
        print("[setup] patched neuralop SpectralConv to keep FFT in fp32 under bf16 autocast", flush=True)

    if args.compile:
        try:
            model = torch.compile(model, mode="default")
            print("[setup] torch.compile mode='default' enabled (avoids cudagraphs/positional-embedding conflict)", flush=True)
        except Exception as exc:
            print(f"[setup] torch.compile failed: {exc} -- continuing without compile", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    h1_loss = H1Loss(d=1, periodic_in_x=True)
    l2_loss = LpLoss(d=1, p=2)

    history = []
    best_full_l2 = math.inf
    t0 = time.time()
    stopped = False
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss, stopped = train_one_epoch(
            model, traj_tr_n, lnu_tr_n, lrho_tr_n, optimizer, h1_loss,
            args.batch_size, device, epoch, t0, args.time_budget_seconds,
            use_amp=args.use_amp, subsample_t=args.subsample_t,
        )
        scheduler.step()

        run_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if run_eval:
            eval_start = time.time()
            h1_step, l2_step, full_l2 = rollout_eval(
                model, traj_te_n, lnu_te_n, lrho_te_n, h1_loss, l2_loss,
                args.rollout_eval_batch, device, use_amp=args.use_amp,
            )
            eval_time = time.time() - eval_start
        else:
            full_l2 = float("nan")
            l2_step = torch.full((traj_te_n.shape[1] - 1,), float("nan"))
            h1_step = torch.full((traj_te_n.shape[1] - 1,), float("nan"))
            eval_time = 0.0

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        if run_eval:
            print(
                f"[epoch {epoch:03d}/{args.epochs}] "
                f"train_h1={train_loss:.4e}  "
                f"rollout_full_l2={full_l2:.4e}  "
                f"l2_step1={l2_step[0].item():.3e}  l2_step50={l2_step[49].item():.3e}  "
                f"l2_step100={l2_step[99].item():.3e}  l2_step200={l2_step[-1].item():.3e}  "
                f"lr={lr_now:.2e}  ep_time={time.time()-ep_start:.0f}s  eval_time={eval_time:.0f}s",
                flush=True,
            )
        else:
            print(
                f"[epoch {epoch:03d}/{args.epochs}] train_h1={train_loss:.4e}  "
                f"lr={lr_now:.2e}  ep_time={time.time()-ep_start:.0f}s (no rollout eval)",
                flush=True,
            )
        history.append({
            "epoch": epoch,
            "train_h1": train_loss,
            "full_l2": full_l2,
            "l2_per_step": l2_step.tolist(),
            "h1_per_step": h1_step.tolist(),
            "lr": lr_now,
            "elapsed": elapsed,
        })

        if run_eval and full_l2 < best_full_l2:
            best_full_l2 = full_l2
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "u_stats": u_stats.to_dict(),
                    "nu_stats": nu_stats.to_dict(),
                    "rho_stats": rho_stats.to_dict(),
                    "epoch": epoch,
                    "full_l2": full_l2,
                    "single_step": True,
                    "residual": True,
                },
                os.path.join(args.ckp_dir, f"{args.run_name}_best.pt"),
            )

        if stopped or elapsed > args.time_budget_seconds:
            print(f"[stop] time budget reached after epoch {epoch}", flush=True)
            break

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "u_stats": u_stats.to_dict(),
            "nu_stats": nu_stats.to_dict(),
            "rho_stats": rho_stats.to_dict(),
            "epoch": epoch,
            "full_l2": full_l2,
            "single_step": True,
            "residual": True,
        },
        os.path.join(args.ckp_dir, f"{args.run_name}_last.pt"),
    )
    with open(os.path.join(args.ckp_dir, f"{args.run_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[done] best_full_rollout_l2={best_full_l2:.4e}  total_time={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
