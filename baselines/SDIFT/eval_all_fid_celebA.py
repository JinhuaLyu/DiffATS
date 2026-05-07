import os
import sys
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image

from FTM_model import Tensor_inr_3D
from networks_edm import UNet



@torch.no_grad()
def edm_sampler_batch(net, latents, t,
                      num_steps=20, sigma_min=0.002, sigma_max=80, rho=7,
                      sigma_data=0.5):
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    i_steps = (sigma_max ** (1 / rho) +
               step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    i_steps = torch.cat([i_steps, torch.zeros_like(i_steps[:1])])

    def forward(x, sigma_scalar):
        sig = torch.full([x.shape[0]], float(sigma_scalar), device=x.device, dtype=torch.float32)
        sig[sig == 0] = sigma_min
        c_skip = sigma_data ** 2 / (sig ** 2 + sigma_data ** 2)
        c_out = sig * sigma_data / (sig ** 2 + sigma_data ** 2).sqrt()
        c_in = 1 / (sigma_data ** 2 + sig ** 2).sqrt()
        c_noise = sig.log() / 4
        scaled = torch.einsum("b,btijk->btijk", c_in, x.float())
        scaled_nchw = scaled.permute(0, 1, 4, 2, 3).contiguous()
        out_nchw = net(scaled_nchw,
                       c_noise.view(-1, 1, 1).repeat(1, t.shape[1], 1),
                       t)
        out = out_nchw.permute(0, 1, 3, 4, 2).contiguous()
        return torch.einsum("b,btijk->btijk", c_skip, x.float()) + \
               torch.einsum("b,btijk->btijk", c_out, out)

    x_next = latents.to(torch.float64) * i_steps[0]
    for i in range(num_steps):
        i_cur = i_steps[i]
        i_next = i_steps[i + 1]
        x_hat = x_next
        denoised = forward(x_hat, i_cur).to(torch.float64)
        d_cur = (x_hat - denoised) / i_cur
        x_next = x_hat + (i_next - i_cur) * d_cur
        if i < num_steps - 1:
            denoised = forward(x_next, i_next).to(torch.float64)
            d_prime = (x_next - denoised) / i_next
            x_next = x_hat + (i_next - i_cur) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next.float()


def build_unet():
    return UNet(
        in_channels=3, out_channels=3,
        num_blocks=2, attn_resolutions=[16],
        model_channels=128, channel_mult=[1, 1, 2, 2, 4, 4],
        dropout=0.10, img_resolution=256,
        label_dim=0, embedding_type="positional",
        encoder_type="standard", decoder_type="standard",
        augment_dim=9, channel_mult_noise=1, resample_filter=[1, 1],
    )


def save_grid(images, path, n=16):
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(n):
        r, c = divmod(i, 4)
        if i < len(images):
            axs[r, c].imshow(images[i])
        axs[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=80, bbox_inches="tight")
    plt.close()


#  Phases
@torch.no_grad()
def phase_generate(args, device, basises, core_mean, core_std):
    """Sample 10k cores via EDM, decode through FTM basis, save PNGs."""
    R = tuple(args.R)
    H = args.image_size

    net = build_unet().to(device)
    net.load_state_dict(torch.load(args.ema_ckpt, map_location=device))
    net.eval()
    n = sum(pp.numel() for pp in net.parameters() if pp.requires_grad)
    print(f"[gen] loaded ema: {args.ema_ckpt}  params={n:,}")

    gen_dir = Path(args.out_dir) / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    grid_collected = []
    n_done = 0
    while n_done < args.num_samples:
        bsz = min(args.gen_batch_size, args.num_samples - n_done)
        shape = (bsz, 1, R[0], R[1], R[2])
        t_grid = torch.linspace(0, 1, shape[1], device=device).view(1, -1, 1).repeat(bsz, 1, 1)
        x_T = torch.randn(shape, device=device)

        cores_norm = edm_sampler_batch(net, x_T, t_grid, num_steps=args.num_steps).detach()
        cores = cores_norm * core_std + core_mean

        out = torch.einsum("mi, btijk->btmjk", basises[0], cores)
        out = torch.einsum("nj, btmjk->btmnk", basises[1], out)
        out = torch.einsum("ok, btmnk->btmno", basises[2], out)
        img = out.squeeze(1).clamp(-1, 1).add(1).mul(127.5)
        img = img.detach().cpu().numpy().astype(np.uint8)

        for j in range(bsz):
            Image.fromarray(img[j]).save(gen_dir / f"{n_done + j:06d}.png")
            if len(grid_collected) < 16:
                grid_collected.append(img[j])

        n_done += bsz
        if n_done % 1000 == 0 or n_done == args.num_samples:
            print(f"  [gen] {n_done}/{args.num_samples}")

    save_grid(grid_collected, Path(args.out_dir) / "visual_grid_gen.png")
    print(f"[gen] saved {n_done} PNGs and visual_grid_gen.png")

    # Free GPU memory before loading 22.6 GB cores file in CPU.
    del net
    torch.cuda.empty_cache()
    return gen_dir


@torch.no_grad()
def phase_reconstruct(args, device, basises):
    """Decode the first 10k FTM cores back to images."""
    print(f"[recon] loading cores from {args.cores_path} (this may take ~30s)")
    cores_cpu = torch.load(args.cores_path, map_location="cpu").float()
    print(f"[recon] cores: shape={tuple(cores_cpu.shape)}  "
          f"mean={float(cores_cpu.mean()):.4f}  std={float(cores_cpu.std()):.4f}")
    n_take = min(args.num_samples, cores_cpu.shape[0])
    print(f"[recon] reconstructing {n_take} cores")

    recon_dir = Path(args.out_dir) / "reconstructed"
    recon_dir.mkdir(parents=True, exist_ok=True)

    grid_collected = []
    n_done = 0
    while n_done < n_take:
        bsz = min(args.decode_batch_size, n_take - n_done)
        idx = torch.arange(n_done, n_done + bsz)
        core_batch = cores_cpu[idx].to(device, non_blocking=True)

        out = torch.einsum("mi, btijk->btmjk", basises[0], core_batch)
        out = torch.einsum("nj, btmjk->btmnk", basises[1], out)
        out = torch.einsum("ok, btmnk->btmno", basises[2], out)
        img = out.squeeze(1).clamp(-1, 1).add(1).mul(127.5)
        img = img.detach().cpu().numpy().astype(np.uint8)

        for j in range(bsz):
            Image.fromarray(img[j]).save(recon_dir / f"{int(idx[j].item()):06d}.png")
            if len(grid_collected) < 16:
                grid_collected.append(img[j])

        n_done += bsz
        if n_done % 1000 == 0 or n_done == n_take:
            print(f"  [recon] {n_done}/{n_take}")

    save_grid(grid_collected, Path(args.out_dir) / "visual_grid_recon.png")
    print(f"[recon] saved {n_done} PNGs and visual_grid_recon.png")
    del cores_cpu
    return recon_dir


def run_fid(label, dir_a, dir_b, out_txt, args):
    # Use sys.executable so the subprocess uses the same conda env (which has
    # pytorch_fid installed) rather than the system /software/miniconda3 python.
    cmd = [
        sys.executable, "-m", "pytorch_fid",
        str(dir_a), str(dir_b),
        "--batch-size", str(args.fid_batch_size),
        "--device", args.device,
    ]
    print(f"[fid:{label}] $ " + " ".join(cmd))
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    print(res.stdout, end="")
    with open(out_txt, "w") as f:
        f.write(f"{label}\n{res.stdout}")
        if res.returncode != 0:
            print(f"[fid:{label}] STDERR:\n{res.stderr}")
            f.write(f"\nSTDERR:\n{res.stderr}")
    return res.stdout.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basis_path", type=str,
                   default="ckp/basis_celebahq_256x256x3_2026_04_28_02.pth")
    p.add_argument("--ema_ckpt", type=str, required=True)
    p.add_argument("--mean_std_mat", type=str, required=True)
    p.add_argument("--cores_path", type=str,
                   default="data/core_celebahq_256x256x3_2026_04_28_02.pt")
    p.add_argument("--real_dir", type=str,
                   default="/home/bkx8728/Tensor_factor/CelebA/CelebA-HQ/celeba_hq_images/all")
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--gen_batch_size", type=int, default=8)
    p.add_argument("--decode_batch_size", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=20, help="EDM sampling steps")
    p.add_argument("--R", type=int, nargs=3, default=(256, 256, 3))
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--fid_batch_size", type=int, default=64)
    p.add_argument("--skip_gen", action="store_true",
                   help="reuse <out_dir>/generated/ if it already exists.")
    p.add_argument("--skip_recon", action="store_true")
    p.add_argument("--skip_fid", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print("device:", device)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- shared: FTM basis + core mean/std ----
    R = tuple(args.R)
    basis = Tensor_inr_3D(R, omega=20).to(device)
    basis.load_state_dict(torch.load(args.basis_path, map_location=device))
    basis.eval()
    basis.mode = "training"
    print(f"loaded basis: {args.basis_path}")

    md = sio.loadmat(args.mean_std_mat)
    core_mean = torch.as_tensor(md["core_mean"]).float().to(device)
    core_std = torch.as_tensor(md["core_std"]).float().to(device)
    print(f"core_mean={float(core_mean):.4f}  core_std={float(core_std):.4f}")

    H = args.image_size
    u_ind = torch.linspace(0, 1, H, device=device)
    v_ind = torch.linspace(0, 1, H, device=device)
    w_ind = torch.linspace(0, 1, 3, device=device)
    with torch.no_grad():
        basises = basis(input_ind_train=(u_ind, v_ind, w_ind))
    print(f"basis matrices: U={tuple(basises[0].shape)}  V={tuple(basises[1].shape)}  W={tuple(basises[2].shape)}")

    # ---- phase 1: generate ----
    gen_dir = out_root / "generated"
    if args.skip_gen and gen_dir.exists() and any(gen_dir.iterdir()):
        print(f"[gen] reusing existing {gen_dir}")
    else:
        gen_dir = phase_generate(args, device, basises, core_mean, core_std)

    # ---- phase 2: reconstruct ----
    recon_dir = out_root / "reconstructed"
    if args.skip_recon and recon_dir.exists() and any(recon_dir.iterdir()):
        print(f"[recon] reusing existing {recon_dir}")
    else:
        recon_dir = phase_reconstruct(args, device, basises)

    # ---- phase 3: three FIDs ----
    if args.skip_fid:
        print("skip_fid set; done.")
        return

    summary = {}
    summary["gen_vs_orig"] = run_fid(
        "gen_vs_orig", gen_dir, args.real_dir, out_root / "fid_gen_vs_orig.txt", args)
    summary["recon_vs_orig"] = run_fid(
        "recon_vs_orig", recon_dir, args.real_dir, out_root / "fid_recon_vs_orig.txt", args)
    summary["recon_vs_gen"] = run_fid(
        "recon_vs_gen", recon_dir, gen_dir, out_root / "fid_recon_vs_gen.txt", args)

    with open(out_root / "fid_summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print("=" * 50)
    print("SUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
