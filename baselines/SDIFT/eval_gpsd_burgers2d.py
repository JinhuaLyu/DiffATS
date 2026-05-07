import argparse
import copy
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from FTM_model import Tensor_inr_3D  # noqa: F401  needed to unpickle basis
from networks_edm import Spatial_temporal_UNet
from karman_dataset import KarmanShardedDataset


def get_gp_covariance(t, gp_gamma=50.0):
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5
    return torch.exp(-torch.square(s) * gp_gamma) + diag


class EDM:
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.model = model.to(self.device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max
        self.rho = cfg.rho
        self.sigma_data = cfg.sigma_data

    def model_forward_wrapper(self, x, sigma, t, cond=None, use_ema=False):
        sigma = sigma.clone()
        sigma[sigma == 0] = self.sigma_min
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out  = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in   = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        net = self.ema if use_ema else self.model
        c_noise_b = c_noise.view(-1, 1, 1).repeat(1, t.shape[1], 1)
        x_scaled = torch.einsum('b,btijk->btijk', c_in, x)
        if cond is not None:
            net_input = torch.cat([x_scaled, cond.to(x_scaled.dtype)], dim=2)
        else:
            net_input = x_scaled
        model_output = net(net_input, c_noise_b, t)
        return (torch.einsum('b,btijk->btijk', c_skip, x)
                + torch.einsum('b,btijk->btijk', c_out, model_output))

    def __call__(self, x, sigma, t, cond=None, use_ema=True):
        if sigma.shape == torch.Size([]):
            sigma = sigma * torch.ones([x.shape[0]]).to(x.device)
        return self.model_forward_wrapper(
            x.float(), sigma.float(), t.float(), cond=cond, use_ema=use_ema)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


@torch.no_grad()
def edm_sampler(edm, latents, t, cond=None, num_steps=250,
                sigma_min=0.002, sigma_max=80.0, rho=7.0, use_ema=True):
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    i_steps = (sigma_max ** (1 / rho)
               + step_indices / (num_steps - 1)
               * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    i_steps = torch.cat([edm.round_sigma(i_steps), torch.zeros_like(i_steps[:1])])
    x_next = latents.to(torch.float64) * i_steps[0]
    for i, (i_cur, i_next) in enumerate(zip(i_steps[:-1], i_steps[1:])):
        x_hat, i_hat = x_next, i_cur
        denoised = edm(x_hat, i_hat, t, cond=cond, use_ema=use_ema).to(torch.float64)
        d_cur = (x_hat - denoised) / i_hat
        x_next = x_hat + (i_next - i_hat) * d_cur
        if i < num_steps - 1:
            denoised = edm(x_next, i_next, t, cond=cond, use_ema=use_ema).to(torch.float64)
            d_prime = (x_next - denoised) / i_next
            x_next = x_hat + (i_next - i_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


# Decoding: Tucker core → spatial field

def decode_core(core, basis_function, device, H=128, W=128):
    """core: (T, R1, R2, R3) → field: (T, 1, H, W)."""
    u_uni = np.array([1.0])
    v_uni = np.linspace(0, H - 1, H) / (H - 1)
    w_uni = np.linspace(0, W - 1, W) / (W - 1)
    u = torch.FloatTensor(u_uni).to(device)
    v = torch.FloatTensor(v_uni).to(device)
    w = torch.FloatTensor(w_uni).to(device)

    basis_function.eval()
    basis_function.mode = "training"
    basises = basis_function(input_ind_train=(u, v, w))
    core = core.to(torch.float32)
    out = torch.einsum("mi, tijk->tmjk", basises[0], core)
    out = torch.einsum("nj, tmjk->tmnk", basises[1], out)
    out = torch.einsum("ok, tmnk->tmno", basises[2], out)
    return out  # (T, 1, H, W)


def encode_t0_core(gt_t0, basis_function, device, H=128, W=128, R=(1, 9, 9)):
    """Least-squares projection of the t=0 pixel frame into Tucker core space.

    Solves: min ||B_H @ C @ B_W.T - Y||_F  where Y=(H,W), C=(R_H,R_W).
    Returns un-normalised core of shape (R_C, R_H, R_W).
    """
    R_C, R_H, R_W = R
    u_uni = torch.FloatTensor(np.array([1.0])).to(device)
    v_uni = torch.FloatTensor(np.linspace(0, H - 1, H) / (H - 1)).to(device)
    w_uni = torch.FloatTensor(np.linspace(0, W - 1, W) / (W - 1)).to(device)

    basis_function.eval()
    basis_function.mode = "training"
    with torch.no_grad():
        basises = basis_function(input_ind_train=(u_uni, v_uni, w_uni))

    B0 = basises[0].float()   # (C=1, R_C=1)
    B1 = basises[1].float()   # (H,   R_H)
    B2 = basises[2].float()   # (W,   R_W)

    # Measurement matrix A: (H*W, R_C*R_H*R_W)
    A = torch.einsum('ci,hj,wk->chwijk', B0, B1, B2).reshape(H * W, R_C * R_H * R_W)

    y = torch.FloatTensor(gt_t0.reshape(-1)).to(device)   # (H*W,)

    AtA = A.T @ A
    AtA += 1e-6 * torch.eye(AtA.shape[0], device=device, dtype=AtA.dtype)
    c_flat = torch.linalg.solve(AtA, A.T @ y)   # (R_C*R_H*R_W,)
    return c_flat.reshape(R_C, R_H, R_W)


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ema_path", required=True)
    p.add_argument("--basis_path", required=True)
    p.add_argument("--core_mean_std_path", required=True)
    p.add_argument("--core_path", required=True,
                   help="FTM core_best.mat — used only to compute DC offset")
    p.add_argument("--data_root",
                   default="${DATA_ROOT}/burgers_2d")
    p.add_argument("--out_dir",
                   default="/projects/e30514/bkx8728/burgers_sdift_runs/eval_cond_t0")
    # Geometry
    p.add_argument("--T", type=int, default=201)
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--W", type=int, default=128)
    p.add_argument("--R", type=int, nargs=3, default=(1, 9, 9))
    # Architecture (must match training)
    p.add_argument("--img_size", type=int, default=9)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--model_channels", type=int, default=208)
    p.add_argument("--channel_mult", type=int, nargs='+', default=[1])
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--num_temporal_latent", type=int, default=8)
    p.add_argument("--attn_resolutions", type=int, nargs='*', default=[])
    # EDM
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--sigma_data", type=float, default=0.5)
    # Sampler
    p.add_argument("--num_steps", type=int, default=250)
    # Eval scope
    p.add_argument("--max_clips", type=int, default=None)
    p.add_argument("--seed", type=int, default=231)
    p.add_argument("--save_every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cond-eval] device={device}  out_dir={out_dir}", flush=True)

    # ---- Model (in_channels=2 for cond_t0, out_channels=1) ----
    in_ch = args.channels * 2
    unet = Spatial_temporal_UNet(
        in_channels=in_ch, out_channels=args.channels,
        num_blocks=args.num_blocks, num_temporal_latent=args.num_temporal_latent,
        attn_resolutions=args.attn_resolutions,
        model_channels=args.model_channels, channel_mult=args.channel_mult,
        dropout=0, img_resolution=args.img_size, label_dim=0,
        embedding_type='positional', encoder_type='standard',
        decoder_type='standard', augment_dim=9, channel_mult_noise=1,
        resample_filter=[1, 1])
    edm = EDM(model=unet, cfg=args)
    state = torch.load(args.ema_path, map_location=device, weights_only=False)
    edm.model.load_state_dict(state)
    edm.ema.load_state_dict(state)
    edm.model.eval()
    edm.ema.eval()
    for p_ in edm.model.parameters():
        p_.requires_grad_(False)
    for p_ in edm.ema.parameters():
        p_.requires_grad_(False)
    n_params = sum(p_.numel() for p_ in edm.model.parameters())
    print(f"[cond-eval] model params={n_params:,}  loaded from {args.ema_path}",
          flush=True)

    # ---- Basis (for decoding sampled cores → field) ----
    basis_function = torch.load(args.basis_path, map_location=device,
                                weights_only=False)
    basis_function.eval()
    for p_ in basis_function.parameters():
        p_.requires_grad_(False)
    print(f"[cond-eval] basis loaded from {args.basis_path}", flush=True)

    # ---- Core normalisation (from GPSD training) ----
    cms = sio.loadmat(args.core_mean_std_path)
    c_mean = float(cms['core_mean'].ravel()[0])
    c_std  = float(cms['core_std'].ravel()[0])
    print(f"[cond-eval] core_mean={c_mean:.5f}  core_std={c_std:.5f}", flush=True)

    # DC offset: FTM joint-optimization cores have a constant shift vs lstsq cores
    # (null space of the SIREN basis).  Align lstsq cores to FTM before normalising.
    core_mat = sio.loadmat(args.core_path)
    ftm_dc = float(np.array(core_mat['core']).mean())
    print(f"[cond-eval] FTM DC offset={ftm_dc:.5f}", flush=True)


    # ---- Test dataset ----
    test_ds = KarmanShardedDataset(args.data_root, split='test',
                                   T=args.T, max_clips=args.max_clips)
    print(f"[cond-eval] N_test={len(test_ds)}  num_steps={args.num_steps}",
          flush=True)

    t_grid = (torch.linspace(0, 1, args.T)
              .view(1, -1, 1).to(device).double())

    rel_l1, rel_l2, rmse_arr = [], [], []
    eps = 1e-8

    N = len(test_ds)
    field_shape = (N, args.T, 1, args.H, args.W)
    pred_mmap = np.memmap(out_dir / "pred_fields.dat", dtype='float32',
                          mode='w+', shape=field_shape)
    gt_mmap   = np.memmap(out_dir / "gt_fields.dat",   dtype='float32',
                          mode='w+', shape=field_shape)
    np.save(out_dir / "field_shape.npy", np.array(field_shape))
    print(f"[cond-eval] saving all {N} pred+gt fields to {out_dir}/pred_fields.dat")

    t0_wall = time.time()

    for idx in range(len(test_ds)):
        clip, _ = test_ds[idx]
        gt_field = clip.numpy()       # (T, 1, H, W)
        gt_t0    = gt_field[0]        # (1, H, W)

        # 1. Encode t=0 frame into Tucker core space via least-squares.
        # lstsq gives cores with mean≈0; FTM joint-optimization cores have mean≈ftm_dc.
        # Adding ftm_dc aligns the lstsq core to the FTM null-space before min-max norm.
        t0_core = encode_t0_core(gt_t0, basis_function, device,
                                 H=args.H, W=args.W, R=args.R)   # (R_C, R_H, R_W)
        t0_core = t0_core + ftm_dc                                 # align DC to FTM
        t0_core_norm = (t0_core - c_mean) / c_std                 # normalise like training

        # 2. Build cond: (1, T, R_C, R_H, R_W)
        cond = (t0_core_norm.unsqueeze(0).unsqueeze(0)
                .expand(1, args.T, -1, -1, -1).double())

        # 3. Sample
        sample_shape = [1, args.T, args.R[0], args.R[1], args.R[2]]
        cov_sample = get_gp_covariance(t_grid)
        L_sample = torch.linalg.cholesky(cov_sample).to(device)
        noise = torch.randn(sample_shape, device=device).double()
        x_T = (L_sample @ noise.view(1, args.T, -1)).view(sample_shape)

        sample = edm_sampler(
            edm, x_T, t_grid, cond=cond,
            num_steps=args.num_steps,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            rho=args.rho, use_ema=True).detach()

        # 4. Unnormalise and decode
        core_sample = (sample[0].float() * c_std + c_mean)  # (T, R1, R2, R3)
        with torch.no_grad():
            field_pred = decode_core(core_sample, basis_function, device,
                                     H=args.H, W=args.W)
        field_pred_np = field_pred.cpu().numpy()  # (T, 1, H, W)

        # 5. Metrics on t=1..T-1
        pred_t = field_pred_np[1:].reshape(-1)
        gt_t   = gt_field[1:].reshape(-1)
        diff   = pred_t - gt_t
        l1  = float(np.abs(diff).sum() / (np.abs(gt_t).sum() + eps))
        l2  = float(np.sqrt((diff ** 2).sum()) / (np.sqrt((gt_t ** 2).sum()) + eps))
        rmse = float(np.sqrt((diff ** 2).mean()))
        rel_l1.append(l1)
        rel_l2.append(l2)
        rmse_arr.append(rmse)

        pred_mmap[idx] = field_pred_np.astype(np.float32)
        gt_mmap[idx]   = gt_field.astype(np.float32)

        elapsed = time.time() - t0_wall
        eta = elapsed / (idx + 1) * (len(test_ds) - idx - 1)
        print(f"[sample {idx+1:4d}/{len(test_ds)}] "
              f"rL1={l1:.5f}  rL2={l2:.5f}  rmse={rmse:.5e}  "
              f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

        if (idx + 1) % args.save_every == 0:
            np.savez(out_dir / "cond_t0_eval_partial.npz",
                     rel_l1=np.array(rel_l1), rel_l2=np.array(rel_l2),
                     rmse=np.array(rmse_arr), n_done=idx + 1)

    rel_l1   = np.array(rel_l1)
    rel_l2   = np.array(rel_l2)
    rmse_arr = np.array(rmse_arr)
    n = len(rel_l1)

    def fmt(arr):
        m   = float(arr.mean())
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        return m, sem

    l1_m,   l1_s   = fmt(rel_l1)
    l2_m,   l2_s   = fmt(rel_l2)
    rmse_m, rmse_s = fmt(rmse_arr)

    print()
    print("=" * 70)
    print(f"Conditional Forecast Eval (N={n})  observe t=0  predict t=1..{args.T-1}")
    print("=" * 70)
    print(f"Average Relative L1 : {l1_m:.4f} ± {l1_s:.4f}")
    print(f"Average Relative L2 : {l2_m:.4f} ± {l2_s:.4f}")
    print(f"Average rMSE        : {rmse_m:.4e} ± {rmse_s:.2e}")
    print("=" * 70)

    np.savez(out_dir / "cond_t0_eval.npz",
             rel_l1=rel_l1, rel_l2=rel_l2, rmse=rmse_arr,
             l1_mean=l1_m, l1_sem=l1_s,
             l2_mean=l2_m, l2_sem=l2_s,
             rmse_mean=rmse_m, rmse_sem=rmse_s,
             n=n)
    pred_mmap.flush()
    gt_mmap.flush()
    print(f"[saved] {out_dir}/cond_t0_eval.npz")
    print(f"[saved] {out_dir}/pred_fields.dat  (shape {field_shape})")
    print(f"[saved] {out_dir}/gt_fields.dat    (shape {field_shape})")


if __name__ == "__main__":
    main()
