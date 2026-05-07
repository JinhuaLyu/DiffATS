from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import sde
from configs import reaction1d_uvit
from datasets_reaction import Reaction1D
from libs.uvit_pde import UViTReaction
from sde_pde import CondScoreModel, euler_maruyama_cond
from DCT_utils_1d import DCT2DBlocks, reverse_zigzag_order_2d, tokens_to_field


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="/scratch/bkx8728/dctdiff_reaction1d/exp_reaction1d/ckpts/280000.ckpt",
        help="path to .ckpt directory containing nnet_ema.pth",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--sample_steps", type=int, default=250)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mixed_precision", default="bf16",
                   choices=["no", "bf16", "fp16"])
    p.add_argument("--max_traj", type=int, default=0,
                   help="limit number of test trajectories (0 = all)")
    p.add_argument("--out_dir",
                   default="/scratch/bkx8728/dctdiff_reaction1d/eval",
                   help="directory to save generated trajectories + metrics")
    p.add_argument("--sampler", default="ode", choices=["ode", "sde"],
                   help="ode (deterministic, used in training viz) or sde (stochastic)")
    return p.parse_args()


def decode_tokens_to_field(tokens, ds):
    """Same decoder as train_reaction.py: tokens -> physical-units field."""
    rev = reverse_zigzag_order_2d(ds.B_t, ds.B_x)
    dct_op = DCT2DBlocks(ds.B_t, ds.B_x).to(tokens.device)
    toks = tokens * ds.Y_bound
    field = tokens_to_field(
        toks, ds.B_t, ds.B_x, ds.M_x, ds.low_freqs, rev,
        n_t=ds.n_t, n_x=ds.n_x, dct_op=dct_op,
    )
    return field * ds.u_scale + ds.shift


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = reaction1d_uvit.get_config()
    device = torch.device(args.device)
    autocast_dtype = {
        "no": None, "bf16": torch.bfloat16, "fp16": torch.float16
    }[args.mixed_precision]

    # --- dataset ---
    dataset = Reaction1D(**cfg.dataset)
    test_ds = dataset.get_split('test')
    if test_ds is None:
        raise RuntimeError("test split not available — check cfg.dataset.test_path")
    print(f"[data] test trajectories: {len(test_ds)}", flush=True)
    print(f"[data] tokens={test_ds.n_tokens} feat={test_ds.feature_dim} "
          f"n_ic_tokens={test_ds.n_ic_tokens}", flush=True)

    # raw ground-truth tensor (avoid re-decoding from DCT tokens, which would
    # introduce floor noise from the fixed compression)
    raw = torch.load(cfg.dataset.test_path, map_location='cpu',
                     weights_only=False, mmap=True)
    u_gt_full = raw['tensor']  # (N, 201, X)
    N_test = len(test_ds)
    if args.max_traj > 0:
        N_test = min(N_test, args.max_traj)
    print(f"[data] evaluating {N_test} trajectories", flush=True)

    # --- model: load EMA weights ---
    nnet = UViTReaction(**cfg.nnet).to(device)
    ema_path = os.path.join(args.ckpt, "nnet_ema.pth")
    print(f"[ckpt] loading EMA weights from {ema_path}", flush=True)
    sd = torch.load(ema_path, map_location='cpu')
    missing, unexpected = nnet.load_state_dict(sd, strict=True)
    nnet.eval()
    n_params = sum(p.numel() for p in nnet.parameters())
    print(f"[model] {n_params/1e6:.4f}M params loaded", flush=True)

    score_model = CondScoreModel(
        nnet, pred=cfg.pred,
        sde=sde.VPSDE(SNR_scale=cfg.dataset.SNR_scale),
        n_ic_tokens=cfg.dataset.n_ic_tokens,
    )

    # --- sampling loop ---
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    L1_list, L2_list, rMSE_list = [], [], []
    u_pred_all = []                                         # (N, 200, X), saved to /scratch
    nu_all, rho_all = [], []
    n_seen = 0
    t_start = time.time()

    autocast_ctx = (
        torch.autocast(device_type='cuda', dtype=autocast_dtype)
        if autocast_dtype is not None else torch.autocast(device_type='cuda', enabled=False)
    )

    for batch_idx, (x_target, nu, rho, ic_mask) in enumerate(test_loader):
        if n_seen >= N_test:
            break
        B = min(x_target.shape[0], N_test - n_seen)
        x_target = x_target[:B].to(device)
        nu = nu[:B].to(device)
        rho = rho[:B].to(device)
        ic_clean = x_target[:, : cfg.dataset.n_ic_tokens]

        x_init = torch.randn_like(x_target)
        rsde = sde.ReverseSDE(score_model) if args.sampler == "sde" else sde.ODE(score_model)
        with torch.no_grad(), autocast_ctx:
            x_gen = euler_maruyama_cond(
                rsde, x_init, sample_steps=args.sample_steps,
                n_ic_tokens=cfg.dataset.n_ic_tokens,
                ic_clean=ic_clean, nu=nu, rho=rho,
            )

        # Decode in fp32 for numerical fidelity of the metrics.
        x_gen_fp32 = x_gen.float()
        field_gen = decode_tokens_to_field(x_gen_fp32, test_ds)  # (B, T_pad, X)
        u_pred = field_gen[
            :, test_ds.gen_start : test_ds.gen_start + 200, :
        ]  # (B, 200, X), physical units

        # Ground truth aligned to this batch.
        u_gt = u_gt_full[n_seen : n_seen + B, 1:201, :].clone().float().to(device)

        diff = u_pred - u_gt                                     # (B, 200, X)
        # Per (traj, t) norms over x-axis.
        eps = 1e-12
        l1_num = diff.abs().sum(dim=-1)                          # (B, 200)
        l1_den = u_gt.abs().sum(dim=-1).clamp_min(eps)
        l2_num = diff.pow(2).sum(dim=-1).sqrt()
        l2_den = u_gt.pow(2).sum(dim=-1).sqrt().clamp_min(eps)
        rmse = diff.pow(2).mean(dim=-1).sqrt()                   # absolute RMSE per (traj,t)

        L1_list.append((l1_num / l1_den).flatten().cpu().numpy())
        L2_list.append((l2_num / l2_den).flatten().cpu().numpy())
        rMSE_list.append(rmse.flatten().cpu().numpy())
        u_pred_all.append(u_pred.detach().cpu())            # (B, 200, X) float32
        nu_all.append(nu.detach().cpu())
        rho_all.append(rho.detach().cpu())

        n_seen += B
        if batch_idx % 4 == 0:
            elapsed = time.time() - t_start
            rate = n_seen / max(elapsed, 1e-6)
            eta = (N_test - n_seen) / max(rate, 1e-6)
            print(f"[eval] {n_seen}/{N_test} traj  ({rate:.1f}/s, eta {eta:.0f}s)",
                  flush=True)

    L1 = np.concatenate(L1_list)
    L2 = np.concatenate(L2_list)
    rMSE = np.concatenate(rMSE_list)

    def fmt(x):
        m = x.mean()
        sem = x.std(ddof=1) / np.sqrt(len(x))
        return f"{m:.5f} ± {sem:.0e}"

    print()
    print("=" * 72)
    print(f"N_traj = {N_test}, N_time = 200, total samples = {len(L1)}")
    print(f"sample_steps = {args.sample_steps}, mixed_precision = {args.mixed_precision}")
    print(f"ckpt = {args.ckpt}")
    print("=" * 72)
    header = "Average Relative Error L1\tAverage Relative Error L2\tAverage rMSE"
    row = f"{fmt(L1)}\t{fmt(L2)}\t{fmt(rMSE)}"
    print(header)
    print(row)
    print("=" * 72)

    # --- save generated trajectories + per-sample metrics to /scratch ---
    os.makedirs(args.out_dir, exist_ok=True)
    u_pred_tensor = torch.cat(u_pred_all, dim=0)            # (N_test, 200, X)
    nu_tensor = torch.cat(nu_all, dim=0)
    rho_tensor = torch.cat(rho_all, dim=0)
    save_path = os.path.join(args.out_dir, "generated.pt")
    torch.save({
        "u_pred": u_pred_tensor,                            # generated u(t=1..200, x)
        "nu": nu_tensor,
        "rho": rho_tensor,
        "t_indices": torch.arange(1, 201, dtype=torch.long),
        "ckpt": args.ckpt,
        "sample_steps": args.sample_steps,
        "mixed_precision": args.mixed_precision,
        "seed": args.seed,
    }, save_path)
    print(f"[save] generated trajectories -> {save_path}  "
          f"shape={tuple(u_pred_tensor.shape)}  "
          f"size={os.path.getsize(save_path)/1e6:.1f} MB")

    metrics_path = os.path.join(args.out_dir, "metrics.pt")
    torch.save({
        "L1_per_sample": L1,                                # (N_test * 200,)
        "L2_per_sample": L2,
        "rMSE_per_sample": rMSE,
        "L1_mean": float(L1.mean()),
        "L1_sem": float(L1.std(ddof=1) / np.sqrt(len(L1))),
        "L2_mean": float(L2.mean()),
        "L2_sem": float(L2.std(ddof=1) / np.sqrt(len(L2))),
        "rMSE_mean": float(rMSE.mean()),
        "rMSE_sem": float(rMSE.std(ddof=1) / np.sqrt(len(rMSE))),
        "n_traj": N_test,
        "n_time": 200,
    }, metrics_path)
    print(f"[save] metrics             -> {metrics_path}")


if __name__ == "__main__":
    main()
