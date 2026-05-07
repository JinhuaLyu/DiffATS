"""Eval for Moving MNIST SDIFT: generate videos for FVD computation.

Two modes (--mode):
  gpsd  (default) — sample cores from GPSD, decode through FTM basis
  ftm             — decode saved Tucker cores directly (FTM ceiling, no GPSD)

Both modes output: dict {'videos': uint8 tensor (N, T, H, W)} saved to --out_pt,
directly consumable by compute_fvd.py.
"""
import argparse
import os

import scipy.io as sio
import torch
from tqdm import tqdm

from train_GPSD import EDM, create_model, edm_sampler, get_gp_covariance


def load_basis(basis_ckpt, H_out, W_out, device):
    basis = torch.load(basis_ckpt, map_location=device, weights_only=False)
    basis.eval()
    basis.mode = "training"
    u_grid = torch.ones(1, dtype=torch.float32, device=device)
    v_grid = torch.linspace(0.0, 1.0, H_out, dtype=torch.float32, device=device)
    w_grid = torch.linspace(0.0, 1.0, W_out, dtype=torch.float32, device=device)
    with torch.no_grad():
        U, V, W = basis(input_ind_train=(u_grid, v_grid, w_grid))
    return U.float(), V.float(), W.float()


def decode_cores(cores, U, V, W):
    """cores: (B, T, R1, R2, R3) -> uint8 (B, T, H, W)."""
    x = torch.einsum("mi,btijk->btmjk", U, cores)
    x = torch.einsum("nj,btmjk->btmnk", V, x)
    x = torch.einsum("ok,btmnk->btmno", W, x)
    return (x.squeeze(2).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()


def run_gpsd(args, device):
    class C: pass
    cfg = C()
    cfg.channels = args.channels
    cfg.img_size = args.img_size
    cfg.model_channels = args.model_channels
    cfg.channel_mult = args.channel_mult
    cfg.attn_resolutions = []
    cfg.layers_per_block = args.layers_per_block
    cfg.num_temporal_latent = args.num_temporal_latent
    cfg.sigma_min = 0.002
    cfg.sigma_max = 80.0
    cfg.rho = 7.0
    cfg.sigma_data = 0.5
    cfg.gt_guide_type = "l2"
    cfg.train_batch_size = args.batch_size
    cfg.device = device

    print(f"Loading GPSD: {args.gpsd_ckpt}")
    mynet = create_model(cfg)
    edm = EDM(model=mynet, cfg=cfg)
    state = torch.load(args.gpsd_ckpt, map_location=device, weights_only=False)
    edm.model.load_state_dict(state)
    edm.ema.load_state_dict(state)
    edm.model.eval(); edm.ema.eval()
    print(f"  params: {sum(p.numel() for p in edm.model.parameters()):,}")

    U, V, W = load_basis(args.basis_ckpt, args.H_out, args.W_out, device)
    print(f"  basis: U={tuple(U.shape)} V={tuple(V.shape)} W={tuple(W.shape)}")

    nm = sio.loadmat(args.core_mean_std_mat)
    core_mean = torch.as_tensor(nm["core_mean"], dtype=torch.float32, device=device).view(())
    core_std  = torch.as_tensor(nm["core_std"],  dtype=torch.float32, device=device).view(())
    print(f"  core_mean={core_mean.item():.4f}  core_std={core_std.item():.4f}")

    R1, R2, R3 = U.shape[1], V.shape[1], W.shape[1]
    N, T, C = args.n_samples, args.T, args.channels
    out = torch.empty((N, T, args.H_out, args.W_out), dtype=torch.uint8)

    print(f"Sampling {N} videos (batch={args.batch_size}, NFE={args.num_steps})")
    with torch.no_grad():
        for i in tqdm(range(0, N, args.batch_size)):
            B = min(args.batch_size, N - i)
            t_grid = torch.linspace(0, 1, T, dtype=torch.float32, device=device).view(1, T, 1).repeat(B, 1, 1)
            cov = get_gp_covariance(t_grid)
            L = torch.linalg.cholesky(cov)
            noise = torch.randn(B, T, C, args.img_size, args.img_size, device=device)
            x_T = (L @ noise.view(B, T, -1)).view(B, T, C, args.img_size, args.img_size)
            sample_norm = edm_sampler(edm, x_T, t_grid, num_steps=args.num_steps).float()
            core = (sample_norm * core_std + core_mean).view(B, T, R1, R2, R3)
            out[i:i + B] = decode_cores(core, U, V, W)
    return out


def run_ftm(args, device):
    U, V, W = load_basis(args.basis_ckpt, args.H_out, args.W_out, device)
    print(f"  basis: U={tuple(U.shape)} V={tuple(V.shape)} W={tuple(W.shape)}")

    import numpy as np
    core_np = sio.loadmat(args.core_mat)["core"]
    core = torch.from_numpy(core_np).float()[:args.n_samples]
    N, T, R1, R2, R3 = core.shape
    print(f"Decoding {N} cores (shape {tuple(core.shape)})")

    out = torch.empty((N, T, args.H_out, args.W_out), dtype=torch.uint8)
    with torch.no_grad():
        for i in tqdm(range(0, N, args.batch_size)):
            B = min(args.batch_size, N - i)
            out[i:i + B] = decode_cores(core[i:i + B].to(device), U, V, W)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gpsd", "ftm"], default="gpsd")
    p.add_argument("--basis_ckpt", required=True)
    p.add_argument("--out_pt", required=True)
    p.add_argument("--n_samples", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--H_out", type=int, default=64)
    p.add_argument("--W_out", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    # GPSD-only
    p.add_argument("--gpsd_ckpt")
    p.add_argument("--core_mean_std_mat")
    p.add_argument("--num_steps", type=int, default=250)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--model_channels", type=int, default=70)
    p.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 2])
    p.add_argument("--layers_per_block", type=int, default=4)
    p.add_argument("--num_temporal_latent", type=int, default=8)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--img_size", type=int, default=32)
    # FTM-only
    p.add_argument("--core_mat")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  mode={args.mode}")

    if args.mode == "gpsd":
        if not args.gpsd_ckpt or not args.core_mean_std_mat:
            p.error("--mode gpsd requires --gpsd_ckpt and --core_mean_std_mat")
        out = run_gpsd(args, device)
    else:
        if not args.core_mat:
            p.error("--mode ftm requires --core_mat")
        out = run_ftm(args, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_pt)), exist_ok=True)
    torch.save({"videos": out}, args.out_pt)
    print(f"Saved {tuple(out.shape)} -> {args.out_pt}")


if __name__ == "__main__":
    main()
