from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from neuralop.models import FNO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",
        default="/gpfs/home/bkx8728/Tensor_factor/1dscripts/fno_reaction/ckp/fno_reaction_1d_ar_b512_best.pt")
    p.add_argument("--test_path",
        default="/scratch/bkx8728/reaction_1d/reaction_1d_test.pt")
    p.add_argument("--save_path",
        default="/scratch/bkx8728/reaction_ar_pred.pt")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_test(path: str):
    print(f"[data] loading {path} ...", flush=True)
    d = torch.load(path, map_location="cpu", weights_only=False)
    full = d["tensor"].float()          # [N, T, X]
    nu   = d["nu"].float()
    rho  = d["rho"].float()
    log_nu  = torch.log10(nu.clamp_min(1e-12))
    log_rho = torch.log10(rho.clamp_min(1e-12))
    print(f"[data] tensor {tuple(full.shape)}", flush=True)
    return full, log_nu, log_rho


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}  torch={torch.__version__}", flush=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    u_stats   = ckpt["u_stats"]
    nu_stats  = ckpt["nu_stats"]
    rho_stats = ckpt["rho_stats"]
    residual  = ckpt.get("residual", False)
    print(f"[ckpt] epoch={ckpt.get('epoch')}  full_l2={ckpt.get('full_l2'):.4e}"
          f"  residual={residual}", flush=True)
    print(f"[ckpt] u_stats  mean={u_stats['mean']:.4f}  std={u_stats['std']:.4f}", flush=True)

    full, log_nu, log_rho = load_test(args.test_path)
    N, T, X = full.shape

    model = FNO(
        n_modes=(train_args["n_modes"],),
        in_channels=3,
        out_channels=1,
        hidden_channels=train_args["hidden_channels"],
        n_layers=train_args["n_layers"],
        channel_mlp_dropout=train_args.get("channel_mlp_dropout", 0.0),
        positional_embedding="grid",
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M", flush=True)

    u_mean   = float(u_stats["mean"])
    u_std    = float(u_stats["std"])
    nu_mean  = float(nu_stats["mean"])
    nu_std   = float(nu_stats["std"])
    rho_mean = float(rho_stats["mean"])
    rho_std  = float(rho_stats["std"])

    n_steps = T - 1   # 200 steps: predict t=1..200

    rel_l1_list, rel_l2_list, rmse_list = [], [], []
    all_pred = []
    eps = 1e-12

    ds     = TensorDataset(full, log_nu, log_rho)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    print(f"[rollout] N={N} trajectories, {n_steps} steps each (t=1..{T-1})", flush=True)
    with torch.no_grad():
        for full_b, lnu_b, lrho_b in loader:
            full_b  = full_b.to(device, non_blocking=True)   # [B, T, X]
            lnu_b   = lnu_b.to(device, non_blocking=True)
            lrho_b  = lrho_b.to(device, non_blocking=True)
            B = full_b.shape[0]

            nu_n  = ((lnu_b  - nu_mean)  / nu_std ).view(B, 1).expand(B, X)
            rho_n = ((lrho_b - rho_mean) / rho_std).view(B, 1).expand(B, X)

            traj_pred = torch.empty(B, T, X, device=device)
            u_curr = full_b[:, 0, :]                          # [B, X] original space
            traj_pred[:, 0, :] = u_curr

            for t in range(n_steps):
                u_n = (u_curr - u_mean) / u_std               # normalize
                x   = torch.stack([u_n, nu_n, rho_n], dim=1)  # [B, 3, X]
                out = model(x).squeeze(1).float()              # [B, X] delta (normalized)
                if residual:
                    u_next_n = u_n + out
                else:
                    u_next_n = out
                u_curr = u_next_n * u_std + u_mean             # decode
                traj_pred[:, t + 1, :] = u_curr

            # metrics over t=1..T-1 vs GT (exclude t=0 which is given)
            pred_steps = traj_pred[:, 1:, :]                  # [B, T-1, X]
            true_steps = full_b[:, 1:, :]                     # [B, T-1, X]
            err      = pred_steps - true_steps
            err_flat = err.flatten(1)                          # [B, (T-1)*X]
            tgt_flat = true_steps.flatten(1)

            l1_rel = err_flat.abs().sum(1) / tgt_flat.abs().sum(1).clamp_min(eps)
            l2_rel = err_flat.pow(2).sum(1).sqrt() / tgt_flat.pow(2).sum(1).sqrt().clamp_min(eps)
            rmse   = err_flat.pow(2).mean(1).sqrt()

            rel_l1_list.append(l1_rel.cpu())
            rel_l2_list.append(l2_rel.cpu())
            rmse_list.append(rmse.cpu())
            all_pred.append(traj_pred.cpu())

    rel_l1 = torch.cat(rel_l1_list)
    rel_l2 = torch.cat(rel_l2_list)
    rmse   = torch.cat(rmse_list)
    preds  = torch.cat(all_pred)   # [N, T, X]

    import math
    n_samp = rel_l1.numel()
    sqrtn  = math.sqrt(n_samp)

    def fmt(t: torch.Tensor) -> str:
        m   = t.mean().item()
        sem = t.std(unbiased=True).item() / sqrtn
        return f"{m:.4g} ± {sem:.2e}"

    l1_str  = fmt(rel_l1)
    l2_str  = fmt(rel_l2)
    rms_str = fmt(rmse)

    print(f"\n[results] N={n_samp} test trajectories, {n_steps} rollout steps")
    print(f"  Average Relative Error L1:  {l1_str}")
    print(f"  Average Relative Error L2:  {l2_str}")
    print(f"  Average rMSE:               {rms_str}")
    print(f"\nTable row (L1 | L2 | rMSE):")
    print(f"  {l1_str}\t{l2_str}\t{rms_str}")

    save_path = args.save_path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"u_pred": preds, "u_true": full}, save_path)
    print(f"\n[saved] predictions -> {save_path}  shape={tuple(preds.shape)}", flush=True)


if __name__ == "__main__":
    main()
