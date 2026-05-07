"""gen_burgers_1d.py — Conditional generation from a trained 1D Burgers Factor-DiT
checkpoint, mirroring `tensor_physics/exp_burgers_2d/generate/gen_burgers_2d.py`.

For each seed in --seeds, iterate the full test dataset (500 samples) with batched
DDIM-respaced sampling. Denormalize the generated patch-SVD factors, also build
the reconstructed physical trajectory ``(NX=1024, T_TRAJ=200)`` per sample, and
dump one ``.pt`` per seed.

Notes
-----
- 1D Burgers v3 was trained with EMA disabled, so ``ckpt["model"]`` is the
  primary state dict (no EMA fall-back).
- Reconstruction matches ``dataset_burgers_1d.reconstruct_traj``: A = alpha @ V_hat^T
  reshaped via the patch grid (32 spatial blocks x 10 time blocks of 32x20 cells).
"""

import argparse
import glob
import os
import sys
import time

import torch

_EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 1d_burgers
sys.path.insert(0, os.path.join(_EXP, "train"))
sys.path.insert(0, "${HOME}/factor_diffusion/video")

from diffusion import create_diffusion                              # noqa: E402

from dataset_burgers_1d import (                                     # noqa: E402
    BurgersFactor1DDataset, reconstruct_traj,
    NX, T_TRAJ,
)
from model_burgers_1d_dit import (                                   # noqa: E402
    build_burgers_1d_dit,
    FLAT_MAIN, FLAT_COND,
    FLAT_ALPHA, FLAT_V_HAT,
    N_MAIN_PATCH, RANK, PATCH_DIM, N_MAIN_RANK,
)


DEFAULT_TRAIN_PATH = ("${DATA_ROOT}/tucker_factors/"
                      "burgers_1d/burgers_1d_train.pt")
DEFAULT_TEST_PATH  = ("${DATA_ROOT}/tucker_factors/"
                      "burgers_1d/burgers_1d_test.pt")
DEFAULT_CKPT_DIR   = ("${DATA_ROOT}/our_method_results/"
                      "burgers_1d/v3/checkpoints")
DEFAULT_OUTDIR     = ("${DATA_ROOT}/our_method_generation/"
                      "burgers_1d")
DEFAULT_STATS_DIR  = ("${DATA_ROOT}/our_method_results/"
                      "burgers_1d/v3/stats")


def resolve_ckpt(ckpt_arg, ckpt_dir, epoch):
    if ckpt_arg:
        return ckpt_arg
    pattern = os.path.join(ckpt_dir, f"epoch{epoch:05d}_step*.pt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint matching {pattern}. Pass --ckpt explicitly."
        )
    return matches[-1]


def pack_test_batch(test_dataset, idxs):
    samples = [test_dataset[i] for i in idxs]
    alpha    = torch.stack([s["alpha"]    for s in samples])
    V_hat    = torch.stack([s["V_hat"]    for s in samples])
    alpha_ic = torch.stack([s["alpha_ic"] for s in samples])
    V_hat_ic = torch.stack([s["V_hat_ic"] for s in samples])
    nu       = torch.stack([s["nu"]       for s in samples])
    sample_idx = torch.tensor([s["idx"] for s in samples], dtype=torch.long)

    x_flat    = torch.cat([alpha.flatten(1),    V_hat.flatten(1)],    dim=1)
    cond_flat = torch.cat([alpha_ic.flatten(1), V_hat_ic.flatten(1)], dim=1)
    return x_flat, cond_flat, nu, sample_idx


def unpack_x(x_flat, B):
    c0, c1 = x_flat.split([FLAT_ALPHA, FLAT_V_HAT], dim=1)
    alpha = c0.reshape(B, N_MAIN_PATCH, RANK)
    V_hat = c1.reshape(B, PATCH_DIM, N_MAIN_RANK)
    return alpha, V_hat


def load_wrapper_from_ckpt(ckpt_path, device):
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg")
    if cfg is None:
        raise RuntimeError("Checkpoint missing `cfg` field.")
    wrapper = build_burgers_1d_dit(cfg).to(device)
    sd = ckpt.get("ema") if ckpt.get("ema") is not None else ckpt["model"]
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    missing, unexpected = wrapper.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
    wrapper.eval()
    print(f"  cfg={cfg}  epoch={ckpt.get('epoch')}  step={ckpt.get('step')}  "
          f"used_state='{'ema' if ckpt.get('ema') is not None else 'model'}'",
          flush=True)
    return wrapper, ckpt


@torch.inference_mode()
def generate_one_seed(seed, wrapper, diffusion_sample, test_dataset,
                     batch_size, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    N = len(test_dataset)
    order = list(range(N))

    alpha_chunks, V_hat_chunks, traj_chunks = [], [], []
    idx_chunks, nu_chunks = [], []

    t_seed = time.time()
    for b0 in range(0, N, batch_size):
        b1 = min(b0 + batch_size, N)
        idxs = order[b0:b1]
        B = len(idxs)

        _x_flat, cond_flat, nu, sample_idx = pack_test_batch(test_dataset, idxs)
        cond_flat = cond_flat.to(device)
        nu = nu.to(device)

        noise = torch.randn(B, FLAT_MAIN, device=device)
        samples = diffusion_sample.p_sample_loop(
            wrapper, noise.shape, noise=noise,
            clip_denoised=False,
            model_kwargs={"cond_flat": cond_flat, "nu": nu},
            device=device, progress=False,
        )
        samples = samples.float()
        alpha_n, V_hat_n = unpack_x(samples, B)

        # Denorm to physical-scale factors
        alpha = test_dataset.denorm(alpha_n, "alpha")
        V_hat = test_dataset.denorm(V_hat_n, "V_hat")

        # Reconstruct trajectory: (B, 1024, 200)
        traj = reconstruct_traj(alpha, V_hat).cpu()

        nu_phys = test_dataset.denorm(nu, "log_nu").exp().cpu()

        alpha_chunks.append(alpha.cpu())
        V_hat_chunks.append(V_hat.cpu())
        traj_chunks.append(traj)
        idx_chunks.append(sample_idx)
        nu_chunks.append(nu_phys)

        print(f"  [seed {seed}]  batch {b0:4d}:{b1:4d}  "
              f"({b1-b0} samples)  elapsed={time.time()-t_seed:.1f}s",
              flush=True)

    return {
        "alpha":      torch.cat(alpha_chunks, dim=0),     # (N, 320, 32)
        "V_hat":      torch.cat(V_hat_chunks, dim=0),     # (N, 640, 32)
        "trajectory": torch.cat(traj_chunks, dim=0),      # (N, 1024, 200)
        "sample_idx": torch.cat(idx_chunks, dim=0),       # (N,)
        "nu":         torch.cat(nu_chunks, dim=0),        # (N,)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",            type=str, default=None)
    parser.add_argument("--ckpt_dir",        type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--epoch",           type=int, default=1000)
    parser.add_argument("--output_dir",      type=str, default=DEFAULT_OUTDIR)
    parser.add_argument("--train_data_path", type=str, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test_data_path",  type=str, default=DEFAULT_TEST_PATH)
    parser.add_argument("--stats_dir",       type=str, default=DEFAULT_STATS_DIR)
    parser.add_argument("--batch_size",      type=int, default=50)
    parser.add_argument("--seeds",           type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--sample_steps",    type=int, default=250)
    parser.add_argument("--noise_schedule",  type=str, default="linear")
    parser.add_argument("--predict_xstart",  action="store_true", default=True)
    parser.add_argument("--T_diffusion",     type=int, default=1000)
    parser.add_argument("--device",          type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ckpt_path = resolve_ckpt(args.ckpt, args.ckpt_dir, args.epoch)
    epoch_tag = f"epoch{args.epoch:05d}"

    # Use train norm-stats so the test set is normalized identically.
    train_dataset = BurgersFactor1DDataset(
        args.train_data_path, stats_dir=args.stats_dir, split="all", device=device,
    )
    test_dataset = BurgersFactor1DDataset(
        args.test_data_path, stats_dir=args.stats_dir, split="all", device=device,
        external_stats=train_dataset.stats,
    )
    del train_dataset
    print(f"Test samples: {len(test_dataset)}", flush=True)

    wrapper, ckpt = load_wrapper_from_ckpt(ckpt_path, device)

    diffusion_sample = create_diffusion(
        timestep_respacing=str(args.sample_steps),
        noise_schedule=args.noise_schedule,
        learn_sigma=False,
        diffusion_steps=args.T_diffusion,
        predict_xstart=args.predict_xstart,
    )
    print(f"Sample steps: {args.sample_steps} (respaced from {args.T_diffusion})  "
          f"schedule={args.noise_schedule}  predict_xstart={args.predict_xstart}",
          flush=True)

    t_total = time.time()
    for seed in args.seeds:
        print(f"\n===== seed {seed} =====", flush=True)
        t0 = time.time()
        out = generate_one_seed(
            seed, wrapper, diffusion_sample, test_dataset, args.batch_size, device,
        )
        out.update({
            "seed":           seed,
            "epoch":          int(ckpt.get("epoch", -1)),
            "step":           int(ckpt.get("step", -1)),
            "ckpt_path":      ckpt_path,
            "sample_steps":   args.sample_steps,
            "noise_schedule": args.noise_schedule,
            "T_diffusion":    args.T_diffusion,
            "predict_xstart": bool(args.predict_xstart),
        })

        out_path = os.path.join(args.output_dir, f"{epoch_tag}_seed{seed}.pt")
        torch.save(out, out_path)
        size_gb = os.path.getsize(out_path) / 1024 ** 3
        print(f"  -> saved {out_path}  ({size_gb:.2f} GB)  "
              f"alpha={tuple(out['alpha'].shape)}  V_hat={tuple(out['V_hat'].shape)}  "
              f"trajectory={tuple(out['trajectory'].shape)}  "
              f"seed_time={time.time()-t0:.1f}s",
              flush=True)
        del out

    print(f"\nAll seeds done in {time.time()-t_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
