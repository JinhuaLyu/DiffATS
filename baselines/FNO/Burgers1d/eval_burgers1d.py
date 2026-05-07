from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from neuralop.models import FNO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/gpfs/home/bkx8728/Tensor_factor/1dscripts/fno_burgers/ckp/fno_burgers_1d_ar_s11_best.pt")
    p.add_argument("--test_path", default="/scratch/bkx8728/burgers_1d/burgers_1d_test.pt")
    p.add_argument("--save_path", default="/scratch/bkx8728/burgers_1d_ar_predictions.pt")
    p.add_argument("--rollout_steps", type=int, default=200,
                   help="number of autoregressive steps to roll out from t=0 (200 -> 201 total slices matching dataset)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)

    print(f"[setup] loading ckpt: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    u_stats = ckpt["u_stats"]
    nu_stats = ckpt["nu_stats"]
    print(f"[setup] ckpt epoch={ckpt.get('epoch')}  full_l2_during_training={ckpt.get('full_l2')}", flush=True)

    model = FNO(
        n_modes=(train_args["n_modes"],),
        in_channels=2,
        out_channels=1,
        hidden_channels=train_args["hidden_channels"],
        n_layers=train_args["n_layers"],
        positional_embedding="grid",
    ).to(device)
    # state_dict may have a "_orig_mod." prefix if the saved model was compiled
    state = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] FNO1d  modes={train_args['n_modes']}  hidden={train_args['hidden_channels']}  layers={train_args['n_layers']}  params={n_params/1e6:.2f}M", flush=True)

    print(f"[data] loading test set: {args.test_path}", flush=True)
    d = torch.load(args.test_path, map_location="cpu", weights_only=False)
    true = d["tensor"].float()                                   # [N, T_gt, X]
    nu = d["nu"].float()
    log_nu = torch.log10(nu.clamp_min(1e-12))
    N, T_gt, X = true.shape
    print(f"[data] {N} trajectories x {T_gt} ground-truth steps x {X} points", flush=True)

    R = args.rollout_steps
    T_pred = R + 1  # includes t=0 input
    print(f"[rollout] generating {R} steps -> trajectory of length {T_pred}", flush=True)

    u_mean, u_std = u_stats["mean"], u_stats["std"]
    nu_mean, nu_std = nu_stats["mean"], nu_stats["std"]

    log_nu_n = ((log_nu - nu_mean) / nu_std).to(device)
    true_dev = true.to(device)

    pred = torch.empty((N, T_pred, X), dtype=torch.float32, device="cpu")
    pred[:, 0, :] = true[:, 0, :]

    for batch_start in range(0, N, args.batch_size):
        batch_end = min(batch_start + args.batch_size, N)
        B = batch_end - batch_start

        u_curr_n = (true_dev[batch_start:batch_end, 0, :] - u_mean) / u_std    # [B, X]
        nu_ch = log_nu_n[batch_start:batch_end].unsqueeze(-1).expand(-1, X)    # [B, X]

        for t in range(R):
            x = torch.stack([u_curr_n, nu_ch], dim=1)                          # [B, 2, X]
            delta = model(x).squeeze(1)                                        # [B, X]
            u_curr_n = u_curr_n + delta
            pred[batch_start:batch_end, t + 1, :] = (u_curr_n * u_std + u_mean).cpu()

        print(f"  rolled out {batch_end}/{N}", flush=True)

    # Metrics: only over the range where ground truth exists, i.e. t in [0, T_gt-1]
    overlap_T = min(T_pred, T_gt)
    print(f"[metrics] computed over the first {overlap_T} time slices (ground-truth length)", flush=True)
    p = pred[:, :overlap_T, :].reshape(N, -1)
    t_ = true[:, :overlap_T, :].reshape(N, -1)
    err = p - t_

    rel_l1 = err.abs().sum(dim=1) / t_.abs().sum(dim=1).clamp_min(1e-12)
    rel_l2 = err.norm(dim=1) / t_.norm(dim=1).clamp_min(1e-12)
    rel_mse = err.pow(2).mean(dim=1) / t_.pow(2).mean(dim=1).clamp_min(1e-12)

    def fmt(v: torch.Tensor) -> str:
        return f"{v.mean().item():.5f} ± {v.std().item():.1e}"

    print()
    print("Average Relative Error L1\tAverage Relative Error L2\tAverage rMSE")
    print(f"{fmt(rel_l1)}\t{fmt(rel_l2)}\t{fmt(rel_mse)}")

    Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pred_traj": pred,                 # [N, T_pred, X]
            "true_traj": true,                 # [N, T_gt, X]
            "nu": nu,                          # [N]
            "rel_l1_per_sample": rel_l1.cpu(),
            "rel_l2_per_sample": rel_l2.cpu(),
            "rel_mse_per_sample": rel_mse.cpu(),
            "rollout_steps": R,
            "T_pred": T_pred,
            "T_gt": T_gt,
            "ckpt_path": args.ckpt,
            "ckpt_epoch": ckpt.get("epoch"),
        },
        args.save_path,
    )
    size_gb = os.path.getsize(args.save_path) / 1e9
    print(f"[save] {args.save_path}  ({size_gb:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
