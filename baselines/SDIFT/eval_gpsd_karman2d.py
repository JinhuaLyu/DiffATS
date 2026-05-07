import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from FTM_model import Tensor_inr_3D
from networks_edm import Spatial_temporal_UNet
from karman_dataset import KarmanShardedDataset, get_grid_coords
from train_GPSD_karman import EDM, edm_sampler, get_gp_covariance


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--basis_path", type=str, required=True)
    p.add_argument("--gpsd_ckpt", type=str, required=True)
    p.add_argument("--core_stats", type=str, required=True,
                   help="core_mean_std.mat from GPSD run (contains global min/max).")
    p.add_argument("--data_root", type=str,
                   default="/projects/p32954/bkx8728/karman_vortex_2d")
    p.add_argument("--T", type=int, default=201)
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--W", type=int, default=128)
    p.add_argument("--R", type=int, nargs=3, default=(1, 12, 12))
    p.add_argument("--omega", type=float, default=20.0)
    # GPSD architecture (must match cond training)
    p.add_argument("--img_size", type=int, default=12)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--model_channels", type=int, default=72)
    p.add_argument("--channel_mult", type=int, nargs='+', default=[1, 2, 2])
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--num_temporal_latent", type=int, default=8)
    p.add_argument("--attn_resolutions", type=int, nargs='*', default=[])
    # EDM
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--sigma_data", type=float, default=0.5)
    p.add_argument("--p_mean", type=float, default=-1.2)
    p.add_argument("--p_std", type=float, default=1.2)
    p.add_argument("--gt_guide_type", type=str, default='l2')
    # Sampling
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--sample_batch_size", type=int, default=4)
    p.add_argument("--sample_steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--save_npz", action="store_true",
                   help="Save generated + GT trajectories as .npz.")
    return p.parse_args()


def decode_cores_to_fields(cores, U, V, W):
    """cores: (B, T, R1, R2, R3) -> fields: (B, T, C, H, W)."""
    out = torch.einsum("mi, btijk->btmjk", U, cores.float())
    out = torch.einsum("nj, btmjk->btmnk", V, out)
    out = torch.einsum("ok, btmnk->btmno", W, out)
    return out


def encode_t0_frame(X0, V_pinv, W_pinv, alpha, device):
    """Encode a single t=0 field to Tucker core via LS projection.

    Args:
        X0:     (H, W) float tensor
        V_pinv: (R2, H) pseudoinverse of V
        W_pinv: (R3, W) pseudoinverse of W
        alpha:  scalar U[0, 0] (channel factor)

    Returns:
        core:   (1, R2, R3) = (R1=1, 12, 12) Tucker core
    """
    # X0[H,W] = alpha * V @ c_mat @ W^T  →  c_mat = (1/alpha) * V_pinv @ X0 @ W_pinv^T
    c_mat = (V_pinv @ X0.to(device) @ W_pinv.t()) / alpha   # (R2, R3)
    return c_mat.unsqueeze(0)                                 # (1, R2, R3)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    print(f"[device] {device}", flush=True)

    # ---- Load FTM basis ----
    basis = torch.load(args.basis_path, map_location=device, weights_only=False)
    if not isinstance(basis, Tensor_inr_3D):
        m = Tensor_inr_3D(tuple(args.R), omega=args.omega).to(device)
        if isinstance(basis, dict):
            m.load_state_dict(basis)
        basis = m
    basis = basis.to(device).eval()
    basis.mode = "training"
    u, v, w = get_grid_coords(H=args.H, W=args.W, device=device)
    with torch.no_grad():
        U, V, W = basis(input_ind_train=(u, v, w))
    U, V, W = U.float(), V.float(), W.float()
    print(f"[basis] U={tuple(U.shape)} V={tuple(V.shape)} W={tuple(W.shape)}", flush=True)

    # Precompute pseudoinverses for encoding t=0 frames.
    # U: (C=1, R1=1) — alpha scalar; V: (H, R2); W: (W, R3)
    alpha = U[0, 0].item()
    V_pinv = torch.linalg.pinv(V)   # (R2=12, H=128)
    W_pinv = torch.linalg.pinv(W)   # (R3=12, W=128)
    print(f"[basis] alpha={alpha:.4f}", flush=True)

    # ---- Load GPSD (cond_t0 model: in_channels = 2*channels) ----
    in_ch = args.channels * 2
    unet = Spatial_temporal_UNet(
        in_channels=in_ch, out_channels=args.channels,
        num_blocks=args.num_blocks,
        num_temporal_latent=args.num_temporal_latent,
        attn_resolutions=args.attn_resolutions,
        model_channels=args.model_channels, channel_mult=args.channel_mult,
        dropout=0, img_resolution=args.img_size,
        label_dim=0, embedding_type='positional',
        encoder_type='standard', decoder_type='standard',
        augment_dim=9, channel_mult_noise=1, resample_filter=[1, 1],
    ).to(device)
    state = torch.load(args.gpsd_ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    # torch.compile wraps keys with '_orig_mod.' — strip if present
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
    unet.load_state_dict(state)
    unet.eval()
    edm = EDM(model=unet, cfg=args)
    edm.ema.load_state_dict(unet.state_dict())
    edm.ema.eval()
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"[gpsd] loaded from {args.gpsd_ckpt}", flush=True)
    print(f"[gpsd] params: {n_params:,}", flush=True)

    # ---- Load core normalization stats ----
    stats = sio.loadmat(args.core_stats)
    c_min = float(np.asarray(stats['core_mean']).ravel()[0])   # "mean" = global min
    c_rng = float(np.asarray(stats['core_std']).ravel()[0])    # "std"  = global range
    print(f"[stats] core_min={c_min:.4f}  core_range={c_rng:.4f}", flush=True)

    # ---- Load test data ----
    test_ds = KarmanShardedDataset(args.data_root, split='test', T=args.T,
                                   max_clips=args.n_samples)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    n_total = min(len(test_ds), args.n_samples)
    print(f"[test] dataset size={len(test_ds)}, evaluating {n_total}", flush=True)

    # ---- Conditional generation + metrics ----
    sum_l1 = sum_l2 = sum_rmse = 0.0
    sum_l1_sq = sum_l2_sq = sum_rmse_sq = 0.0
    n_done = 0
    save_recons, save_gts = [], []
    gen_t0 = time.time()

    for batch_idx, (clip_batch, _) in enumerate(test_loader):
        if n_done >= n_total:
            break

        clip = clip_batch[0].float()   # (T=201, 1, H=128, W=128)

        # 1. Encode t=0 frame to Tucker core
        X0 = clip[0, 0, :, :]         # (H, W)
        with torch.no_grad():
            c0_raw = encode_t0_frame(X0, V_pinv, W_pinv, alpha, device)  # (1, 12, 12)
        c0_norm = (c0_raw - c_min) / c_rng                               # normalized

        # 2. Build conditioning tensor: (1, T, 1, 12, 12)
        cond = c0_norm.unsqueeze(0).unsqueeze(0).expand(1, args.T, args.channels,
                                                         args.img_size, args.img_size)
        cond = cond.contiguous().to(device)

        # 3. Sample from EDM
        t_grid = torch.linspace(0, 1, args.T).view(1, -1, 1).to(device)
        cov = get_gp_covariance(t_grid)
        L_chol = torch.linalg.cholesky(cov)
        noise = torch.randn(1, args.T, args.channels,
                            args.img_size, args.img_size).to(device)
        x_T = (L_chol @ noise.view(1, args.T, -1)).view(
            1, args.T, args.channels, args.img_size, args.img_size)

        with torch.no_grad():
            cores_norm = edm_sampler(edm, x_T, t_grid, cond=cond,
                                     num_steps=args.sample_steps,
                                     sigma_min=args.sigma_min,
                                     sigma_max=args.sigma_max,
                                     rho=args.rho).detach()   # (1, T, 1, 12, 12)

        # 4. Un-normalize and decode
        cores = cores_norm.float() * c_rng + c_min             # (1, T, 1, 12, 12)
        with torch.no_grad():
            fields_gen = decode_cores_to_fields(cores, U, V, W)  # (1, T, 1, H, W)
        fields_gen = fields_gen[0].cpu()   # (T, 1, H, W)

        # 5. Metrics on t=1..200 (exclude given t=0)
        gt = clip[1:].float()            # (200, 1, H, W) on CPU
        gen = fields_gen[1:]             # (200, 1, H, W)
        diff = gen - gt
        gt_l1 = gt.abs().sum().item()
        gt_l2 = gt.norm().item()
        diff_l1 = diff.abs().sum().item()
        diff_l2 = diff.norm().item()

        rel_l1 = diff_l1 / (gt_l1 + 1e-10)
        rel_l2 = diff_l2 / (gt_l2 + 1e-10)
        rmse_n = (diff_l2 ** 2) / (gt_l2 ** 2 + 1e-20)

        sum_l1 += rel_l1
        sum_l2 += rel_l2
        sum_rmse += rmse_n
        sum_l1_sq += rel_l1 ** 2
        sum_l2_sq += rel_l2 ** 2
        sum_rmse_sq += rmse_n ** 2

        if args.save_npz and n_done < 64:
            save_recons.append(fields_gen.numpy())
            save_gts.append(clip.float().numpy())

        n_done += 1
        if n_done % 20 == 0 or n_done == 1:
            elapsed = time.time() - gen_t0
            rl2_so_far = sum_l2 / n_done
            print(f"  [{n_done}/{n_total}]  elapsed={elapsed:.1f}s  "
                  f"per_sample={elapsed/n_done:.1f}s  "
                  f"running_rel_l2={rl2_so_far:.4f}", flush=True)

    # ---- Final metrics ----
    avg_l1 = sum_l1 / n_done
    avg_l2 = sum_l2 / n_done
    avg_rmse = sum_rmse / n_done
    std_l1 = max(sum_l1_sq / n_done - avg_l1 ** 2, 0.0) ** 0.5
    std_l2 = max(sum_l2_sq / n_done - avg_l2 ** 2, 0.0) ** 0.5
    std_rmse = max(sum_rmse_sq / n_done - avg_rmse ** 2, 0.0) ** 0.5

    print()
    print("=" * 64)
    print(f"  Conditional SDIFT (given t=0 → generate t=1..200)")
    print(f"  n = {n_done}, sample_steps = {args.sample_steps}")
    print(f"  Average Relative Error L1 : {avg_l1:.5f} ± {std_l1:.1e}")
    print(f"  Average Relative Error L2 : {avg_l2:.5f} ± {std_l2:.1e}")
    print(f"  Average rMSE              : {avg_rmse:.5f} ± {std_rmse:.1e}")
    print("=" * 64)

    if args.out_dir:
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "metrics_cond.txt", "w") as f:
            f.write(f"n = {n_done}\n")
            f.write(f"sample_steps = {args.sample_steps}\n")
            f.write(f"Average Relative Error L1 : {avg_l1:.5f} ± {std_l1:.1e}\n")
            f.write(f"Average Relative Error L2 : {avg_l2:.5f} ± {std_l2:.1e}\n")
            f.write(f"Average rMSE              : {avg_rmse:.5f} ± {std_rmse:.1e}\n")
        if args.save_npz and save_recons:
            np.savez_compressed(str(out_path / "cond_gen_samples.npz"),
                                generated=np.stack(save_recons, axis=0),
                                ground_truth=np.stack(save_gts, axis=0))
        print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
