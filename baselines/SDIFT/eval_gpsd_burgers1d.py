
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

from networks_edm_1d import Spatial_temporal_UNet_1D
from train_GPSD_1d import EDM, edm_sampler
from utils_1d import get_gp_covariance


def per_sample_metrics(pred: np.ndarray, gt: np.ndarray):
    """pred, gt : (n, T, X). Returns per-sample arrays of length n."""
    eps = 1e-12
    diff = pred - gt
    n = pred.shape[0]
    rmse = np.sqrt(np.mean(diff.reshape(n, -1) ** 2, axis=1))
    rng = (gt.reshape(n, -1).max(axis=1) - gt.reshape(n, -1).min(axis=1)).clip(eps)
    nrmse = rmse / rng
    rel_l1 = (
        np.abs(diff.reshape(n, -1)).sum(axis=1)
        / np.abs(gt.reshape(n, -1)).sum(axis=1).clip(eps)
    )
    rel_l2 = (
        np.linalg.norm(diff.reshape(n, -1), axis=1)
        / np.linalg.norm(gt.reshape(n, -1), axis=1).clip(eps)
    )
    return rmse, nrmse, rel_l1, rel_l2


def fmt_pm(mean: float, sem: float) -> str:
    if sem == 0 or not np.isfinite(sem):
        return f"{mean:.4f} ± 0"
    # SEM compact: e.g. 1e-4
    exp = int(np.floor(np.log10(sem))) if sem > 0 else 0
    coef = sem / 10 ** exp
    if abs(coef - round(coef)) < 0.05:
        sem_str = f"{int(round(coef))}e{exp}"
    else:
        sem_str = f"{coef:.1f}e{exp}"
    return f"{mean:.4f} ± {sem_str}"


def report_block(name: str, rmse, nrmse, rel_l1, rel_l2):
    n = len(rmse)
    sem = lambda x: np.std(x, ddof=1) / np.sqrt(n) if n > 1 else 0.0
    # Three metrics in the order the user requested: avg Rel-L1, avg Rel-L2, avg RMSE.
    return (
        f"[{name:<6s}  n={n:4d}]  "
        f"avg Rel-L1 err {fmt_pm(rel_l1.mean(), sem(rel_l1))}    "
        f"avg Rel-L2 err {fmt_pm(rel_l2.mean(), sem(rel_l2))}    "
        f"avg RMSE       {fmt_pm(rmse.mean(), sem(rmse))}"
    )


# Loaders.
def load_basis(path, device):
    basis = torch.load(path, map_location=device, weights_only=False)
    basis.eval()
    basis.mode = "training"
    return basis


def load_norm(core_norm_path):
    cn = sio.loadmat(core_norm_path)
    return (
        float(np.asarray(cn["core_min"]).reshape(-1)[0]),
        float(np.asarray(cn["core_range"]).reshape(-1)[0]),
    )


def load_cond_meta(cond_meta_path, r1_total):
    """r1_total = patch_r1 * r1_resolution (the unpatched FTM core rank)."""
    cmeta = sio.loadmat(cond_meta_path)
    log_nu_mean = float(np.asarray(cmeta["log_nu_mean"]).reshape(-1)[0])
    log_nu_std = float(np.asarray(cmeta["log_nu_std"]).reshape(-1)[0])
    cond_dim = int(np.asarray(cmeta["cond_dim"]).reshape(-1)[0])
    has_rho = "log_rho_mean" in cmeta
    log_rho_mean = float(np.asarray(cmeta["log_rho_mean"]).reshape(-1)[0]) if has_rho else None
    log_rho_std = float(np.asarray(cmeta["log_rho_std"]).reshape(-1)[0]) if has_rho else None
    n_scalars = 2 if has_rho else 1
    expected = n_scalars + r1_total
    assert cond_dim == expected, f"cond_dim mismatch: {cond_dim} vs {expected}"
    return dict(log_nu_mean=log_nu_mean, log_nu_std=log_nu_std,
                log_rho_mean=log_rho_mean, log_rho_std=log_rho_std,
                has_rho=has_rho, cond_dim=cond_dim, n_scalars=n_scalars)


def build_edm(cfg, cond_dim, device):
    net = Spatial_temporal_UNet_1D(
        r1_resolution=cfg.r1_resolution,
        in_channels=cfg.channels,
        out_channels=cfg.channels,
        model_channels=cfg.model_channels,
        channel_mult=cfg.channel_mult,
        num_blocks=cfg.layers_per_block,
        num_temporal_latent=cfg.num_temporal_latent,
        attn_resolutions=cfg.attn_resolutions,
        dropout=0.0,
        cond_dim=cond_dim,
    ).to(device)
    return EDM(model=net, cfg=cfg)


def load_edm_weights(edm, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    edm.model.load_state_dict(sd)
    edm.ema.load_state_dict(sd)
    edm.model.eval(); edm.ema.eval()
    for p in edm.model.parameters(): p.requires_grad = False
    for p in edm.ema.parameters(): p.requires_grad = False


# Per-method runners.
@torch.no_grad()
def run_cond(cfg, basis, edm_cond, U, U_pinv, core_min, core_range, cmeta,
             tensor_te, nu_te, rho_te, n_run, device):
    T, X = tensor_te.shape[1], tensor_te.shape[2]
    R1_eff = cfg.r1_resolution                 # what the U-Net sees (patched)
    R1_total = cfg.patch_r1 * R1_eff           # original FTM rank
    t_grid_template = torch.linspace(0, 1, T, device=device).view(1, -1, 1)

    preds = np.zeros((n_run, T, X), dtype=np.float32)
    gts = np.zeros((n_run, T, X), dtype=np.float32)

    for i in tqdm(range(n_run), desc="cond"):
        u_tx_gt = tensor_te[i].to(device, non_blocking=True).clone()  # (T, X)
        nu_i = float(nu_te[i].item())
        rho_i = float(rho_te[i].item()) if cmeta["has_rho"] else None

        # Encode IC at the original (unpatched) rank.
        c0_phys = (U_pinv @ u_tx_gt[0])         # (R1_total,)
        c0_norm = (c0_phys - core_min) / core_range  # (R1_total,)

        feats = [(np.log10(max(nu_i, 1e-12)) - cmeta["log_nu_mean"])
                 / (cmeta["log_nu_std"] + 1e-8)]
        if cmeta["has_rho"]:
            feats.append((np.log10(max(rho_i, 1e-12)) - cmeta["log_rho_mean"])
                         / (cmeta["log_rho_std"] + 1e-8))
        cond_vec = torch.cat([
            torch.tensor(feats, device=device, dtype=torch.float32),
            c0_norm.float()
        ]).unsqueeze(0)  # (1, n_scalars + R1_total)

        t_grid = t_grid_template.repeat(1, 1, 1)
        cov = get_gp_covariance(t_grid)
        L = torch.linalg.cholesky(cov)
        noise = torch.randn(1, T, cfg.channels, R1_eff, device=device)
        x_T = (L @ noise.view(1, T, -1)).view(1, T, cfg.channels, R1_eff)
        # IC clamp: reshape (R1_total,) -> (channels=patch_r1, R1_eff)
        ic_clamp = c0_norm.float().view(1, cfg.patch_r1, R1_eff)
        samp = edm_sampler(edm_cond, x_T, t_grid, num_steps=cfg.total_steps,
                           cond=cond_vec, ic_clamp=ic_clamp).detach()

        # Unpatch: (T, channels=patch_r1, R1_eff) -> (T, R1_total)
        core_pred = samp[0].float().reshape(T, R1_total) * core_range + core_min
        u_pred = torch.einsum("xr, tr -> tx", U, core_pred)
        u_pred[0] = u_tx_gt[0]
        preds[i] = u_pred.cpu().numpy()
        gts[i] = u_tx_gt.cpu().numpy()
    return preds, gts



# Main.
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--basis_path", required=True)
    p.add_argument("--test_pt", default="/scratch/bkx8728/burgers_1d/burgers_1d_test.pt")
    p.add_argument("--out_dir", default="/scratch/bkx8728/sdift_1d_runs/results")
    p.add_argument("--out_tag", default="eval")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--total_steps", type=int, default=30)
    p.add_argument("--cond_norm_path", required=True)
    p.add_argument("--cond_meta_path", required=True)
    p.add_argument("--cond_ckpt_path", required=True)
    p.add_argument("--n_cond", type=int, default=500)
    # Architecture.
    p.add_argument("--r1_resolution", type=int, default=152)
    p.add_argument("--patch_r1", type=int, default=1,
                   help="Spatial patch factor; must match training (1 = no patching).")
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--model_channels", type=int, default=40)
    p.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 2, 2])
    p.add_argument("--attn_resolutions", type=int, nargs="+", default=[16, 8])
    p.add_argument("--layers_per_block", type=int, default=2)
    p.add_argument("--num_temporal_latent", type=int, default=4)
    # EDM.
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--sigma_data", type=float, default=0.5)
    p.add_argument("--gt_guide_type", default="l2")
    return p.parse_args()


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"device={device}  out_dir={cfg.out_dir}")

    basis = load_basis(cfg.basis_path, device)
    d = torch.load(cfg.test_pt, map_location="cpu", weights_only=False, mmap=True)
    tensor_te = d["tensor"]                # (N_te, T, X)
    nu_te = d["nu"].float()
    rho_te = d["rho"].float() if "rho" in d else None
    x_coord = d["x_coord"].float().to(device)
    N_te, T, X = tensor_te.shape
    print(f"test set: N={N_te}, T={T}, X={X}")

    with torch.no_grad():
        U = basis(x_coord)                 # (X, R1)
        U_pinv = torch.linalg.pinv(U)      # (R1, X)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    summary_path = os.path.join(cfg.out_dir, f"eval_summary_{cfg.out_tag}_{ts}.txt")

    cmeta = load_cond_meta(cfg.cond_meta_path, cfg.patch_r1 * cfg.r1_resolution)
    core_min, core_range = load_norm(cfg.cond_norm_path)
    edm_cond = build_edm(cfg, cmeta["cond_dim"], device)
    load_edm_weights(edm_cond, cfg.cond_ckpt_path, device)

    n_run = min(cfg.n_cond, N_te)
    t0 = time.time()
    preds, gts = run_cond(cfg, basis, edm_cond, U, U_pinv, core_min, core_range,
                          cmeta, tensor_te, nu_te, rho_te, n_run, device)
    elapsed = time.time() - t0
    rmse, nrmse, rel_l1, rel_l2 = per_sample_metrics(preds, gts)
    line = report_block("cond", rmse, nrmse, rel_l1, rel_l2)
    print(line)

    out_path = os.path.join(cfg.out_dir, f"eval_cond_{cfg.out_tag}_{ts}.npz")
    np.savez_compressed(
        out_path,
        preds=preds, gts=gts,
        rmse=rmse, nrmse=nrmse, rel_l1=rel_l1, rel_l2=rel_l2,
        nu=nu_te[:n_run].numpy(),
        x_coord=x_coord.cpu().numpy(),
        n=n_run, ckpt=cfg.cond_ckpt_path,
    )
    print(f"saved {out_path}")

    with open(summary_path, "w") as f:
        f.write(line + f"   ({elapsed:.1f}s)\n")
        f.write(f"saved: {out_path}\n")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
