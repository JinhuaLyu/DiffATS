"""
Generic 10k-sample driver for all 4 ablation methods.

For a given METHOD in {our_method, shared_bases, no_alignment, data_augmentation}:
  1. Find latest experiment_dir under {results_root}/{method}/, locate final.pt
     (or the latest epoch_*.pt if final.pt not present yet).
  2. Build the right model class, load weights (strict, after stripping _orig_mod.).
  3. Build a DDIM diffusion with sample_steps (default 250).
  4. Resume: count existing PNGs in {samples_dir}/images/ and skip ahead.
  5. Sample NUM_SAMPLES=10000 latents in batches of BATCH_SIZE.
     For every SHARD_SIZE=500 latents, write one shard:
       {samples_dir}/latents/shard_NNNN.pt = {"latents": (B,..), "indices": [...]}
     For every batch, decode -> save individual PNGs as {idx:05d}.png.

Reconstruction (per method):
  our_method        : alpha @ V_hat.T + mean_ref          (per-sample V_hat from diffusion output)
  shared_bases      : alpha @ D.T     + global_mean
  no_alignment      : alpha @ V_hat.T + mean_avg          (per-sample V_hat from diffusion output)
  data_augmentation : same as no_alignment
"""

from __future__ import annotations
import argparse
import glob
import os
import re
import sys
import time
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.amp import autocast


# Paths
_HERE       = os.path.dirname(os.path.abspath(__file__))           # .../celeba_hq/methods
_CELEBA_HQ  = os.path.dirname(_HERE)                               # .../celeba_hq
_OUR_METHOD = os.path.join(_HERE, "our_method")
_RESULTS    = "/anvil/projects/x-eng260004/factor_diffusion/ablation_results"
_FACTORS    = "/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/celeba"


# ---------- Imports from our_method (without modifying it) ----------
sys.path.insert(0, _OUR_METHOD)
sys.path.insert(0, _HERE)        # for shared_bases dit_models_alpha_only
sys.path.insert(0, _CELEBA_HQ)   # for diffusion package
from dit_models import JointDiT  # noqa: E402
from diffusion import create_diffusion  # noqa: E402

_om_spec  = importlib.util.spec_from_file_location("our_method_train", os.path.join(_OUR_METHOD, "train.py"))
_om_train = importlib.util.module_from_spec(_om_spec)
sys.modules["our_method_train"] = _om_train
_om_spec.loader.exec_module(_om_train)
recover_from_alpha_to_image = _om_train.recover_from_alpha_to_image
infer_joint_layout          = _om_train.infer_joint_layout


def _strip_compile_prefix(sd: dict) -> dict:
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


# ---------- Find checkpoint ----------
def find_latest_ckpt(results_dir: str) -> str:
    """Latest epoch_*.pt or final.pt under any sub-experiment dir."""
    finals = sorted(glob.glob(os.path.join(results_dir, "*", "checkpoints", "final.pt")))
    if finals:
        return finals[-1]
    epochs = sorted(glob.glob(os.path.join(results_dir, "*", "checkpoints", "epoch_*.pt")))
    if epochs:
        return epochs[-1]
    raise FileNotFoundError(f"No checkpoint under {results_dir}/*/checkpoints/")


# ---------- Resume / PNG accounting ----------
_PNG_RE = re.compile(r"^(\d{5})\.png$")

def count_existing_pngs(images_dir: str) -> int:
    if not os.path.isdir(images_dir):
        return 0
    idxs = []
    for n in os.listdir(images_dir):
        m = _PNG_RE.match(n)
        if m:
            idxs.append(int(m.group(1)))
    if not idxs:
        return 0
    idxs.sort()
    for expected, got in enumerate(idxs):
        if expected != got:
            raise RuntimeError(
                f"PNG files not contiguous from 00000.png in {images_dir}; expected {expected:05d}, got {got:05d}.png"
            )
    return len(idxs)


# ---------- PNG saving ----------
def _save_png(args):
    arr_uint8_hwc, path = args
    Image.fromarray(arr_uint8_hwc).save(path, format="PNG", compress_level=1)


def save_pngs_async(images_chw_float: torch.Tensor, start_idx: int, images_dir: str,
                    executor: ThreadPoolExecutor) -> list:
    arr = (images_chw_float.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    futures = []
    for i in range(arr.shape[0]):
        p = os.path.join(images_dir, f"{start_idx + i:05d}.png")
        futures.append(executor.submit(_save_png, (arr[i], p)))
    return futures


# ---------- Method-specific decoders ----------
def _decode_joint_with_mean(samples: torch.Tensor, alpha_n: int,
                            alpha_std: torch.Tensor, vhat_std: float,
                            mean_per_channel: torch.Tensor,
                            patch: int, img_hw: int) -> torch.Tensor:
    """For our_method / no_alignment / data_augmentation. samples=(B,3,joint_h,R)."""
    alpha = samples[:, :, :alpha_n, :].float()
    Vhat  = samples[:, :, alpha_n:, :].float()
    alpha = alpha * alpha_std[None, :, None, :]
    Vhat  = Vhat * vhat_std
    return recover_from_alpha_to_image(alpha, Vhat, mean_per_channel, patch=patch, img_hw=img_hw)


def _decode_alpha_with_global_dict(samples: torch.Tensor,
                                   alpha_std: torch.Tensor,
                                   D: torch.Tensor, mean: torch.Tensor,
                                   patch: int, img_hw: int) -> torch.Tensor:
    """For shared_bases. samples=(B,3,alpha_n,R)."""
    alpha = samples.float() * alpha_std[None, :, None, :]
    return recover_from_alpha_to_image(alpha, D, mean, patch=patch, img_hw=img_hw)


# ---------- Main per-method ----------
def run(method: str, num_samples: int = 10000, batch_size: int = 32,
        sampler: str = "ddim", num_sampling_steps: int = 250, ddim_eta: float = 0.0,
        seed: int = 42, ckpt: Optional[str] = None, samples_subdir: str = "samples"):
    assert method in ("our_method", "shared_bases", "no_alignment", "data_augmentation"), method
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output paths
    results_dir = os.path.join(_RESULTS, method)
    samples_dir = os.path.join(results_dir, samples_subdir)
    images_dir  = os.path.join(samples_dir, "images")
    latents_dir = os.path.join(samples_dir, "latents")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(latents_dir, exist_ok=True)

    print(f"[INFO] method = {method}")
    print(f"[INFO] device = {device}")
    print(f"[INFO] num_samples = {num_samples}, batch_size = {batch_size}")
    print(f"[INFO] sampler = {sampler}, steps = {num_sampling_steps}, eta = {ddim_eta}")
    print(f"[INFO] samples_dir = {samples_dir}")

    # Locate ckpt
    ckpt_path = ckpt if ckpt else find_latest_ckpt(results_dir)
    print(f"[INFO] checkpoint = {ckpt_path}")
    try:
        torch.serialization.add_safe_globals([argparse.Namespace])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", None)
    if train_args is None:
        raise RuntimeError(f"checkpoint at {ckpt_path} has no 'args'")

    # Architecture knobs (always taken from training-time args)
    hidden_size = getattr(train_args, "hidden_size", 1152)
    depth       = getattr(train_args, "depth", 28)
    num_heads   = getattr(train_args, "num_heads", 16)
    mlp_ratio   = getattr(train_args, "mlp_ratio", 4.0)
    rank        = getattr(train_args, "svd_rank", 32)
    img_hw      = getattr(train_args, "image_size", 1024)
    patch       = getattr(train_args, "patch_size", [32, 32])
    patch       = patch[0] if isinstance(patch, (list, tuple)) else int(patch)
    alpha_n     = (img_hw // patch) ** 2
    patch_dim   = patch * patch
    print(f"[INFO] arch hidden={hidden_size} depth={depth} heads={num_heads} mlp={mlp_ratio} "
          f"rank={rank} patch={patch} img={img_hw}")

    # Build model
    if method == "shared_bases":
        sys.path.insert(0, os.path.join(_HERE, "shared_bases"))
        from dit_models_alpha_only import AlphaOnlyDiT
        model = AlphaOnlyDiT(
            hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            img_size=img_hw, patch_size=patch, rank=rank,
        )
    else:
        model = JointDiT(
            hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            img_size=img_hw, patch_size=patch, rank=rank,
        )
    sd = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(_strip_compile_prefix(sd), strict=True)
    model = model.to(device).eval()

    # Stats + reconstruction inputs (per method)
    if method == "our_method":
        stats_dir = os.path.join(_FACTORS, "our_method")
        alpha_std = torch.load(os.path.join(stats_dir, "alpha_stats_procrustes_refimg_p32_r32.pt"),
                               map_location="cpu", weights_only=False)["std"].float().to(device)
        vhat_std  = torch.load(os.path.join(stats_dir, "vhat_stats_procrustes_refimg_p32_r32.pt"),
                               map_location="cpu", weights_only=False)["std"].float().item()
        anchor    = torch.load(os.path.join(stats_dir, "ref_anchor.pt"),
                               map_location="cpu", weights_only=False)
        mean_pc   = anchor["mean_ref"].float().to(device)              # (3, patch_dim)
        decode = lambda s: _decode_joint_with_mean(s, alpha_n, alpha_std, vhat_std, mean_pc, patch, img_hw)
        z_shape = (3, alpha_n + patch_dim, rank)
    elif method == "shared_bases":
        stats_dir = os.path.join(_FACTORS, "shared_bases")
        alpha_std = torch.load(os.path.join(stats_dir, "alpha_stats_global_pca_p32_r32.pt"),
                               map_location="cpu", weights_only=False)["std"].float().to(device)
        dct       = torch.load(os.path.join(stats_dir, "global_dict.pt"),
                               map_location="cpu", weights_only=False)
        D         = dct["D"].float().to(device)                        # (3, patch_dim, R)
        mean_pc   = dct["mean"].float().to(device)                     # (3, patch_dim)
        decode = lambda s: _decode_alpha_with_global_dict(s, alpha_std, D, mean_pc, patch, img_hw)
        z_shape = (3, alpha_n, rank)
    else:  # no_alignment / data_augmentation
        stats_dir = os.path.join(_FACTORS, "no_alignment")
        alpha_std = torch.load(os.path.join(stats_dir, "alpha_stats_no_alignment_p32_r32.pt"),
                               map_location="cpu", weights_only=False)["std"].float().to(device)
        vhat_std  = torch.load(os.path.join(stats_dir, "vhat_stats_no_alignment_p32_r32.pt"),
                               map_location="cpu", weights_only=False)["std"].float().item()
        mean_pc   = torch.load(os.path.join(stats_dir, "mean_avg_no_alignment_p32_r32.pt"),
                               map_location="cpu", weights_only=False)["mean_avg"].float().to(device)
        decode = lambda s: _decode_joint_with_mean(s, alpha_n, alpha_std, vhat_std, mean_pc, patch, img_hw)
        z_shape = (3, alpha_n + patch_dim, rank)

    # Diffusion
    timestep_respacing = f"ddim{num_sampling_steps}" if sampler == "ddim" else str(num_sampling_steps)
    diffusion = create_diffusion(
        timestep_respacing,
        learn_sigma=getattr(train_args, "learn_sigma", False),
        predict_xstart=True,
    )

    # Resume
    n_done = count_existing_pngs(images_dir)
    print(f"[INFO] existing PNGs = {n_done}")
    if n_done >= num_samples:
        print(f"[INFO] already have {n_done} >= {num_samples} samples; nothing to do.")
        return

    # Sampling loop
    SHARD_SIZE = 500
    n_thread   = min(8, os.cpu_count() or 4)
    print(f"[INFO] threads for PNG save = {n_thread}")

    # Buffer that flushes a shard every SHARD_SIZE latents
    shard_buf_lat: list[torch.Tensor] = []
    shard_buf_idx: list[int] = []
    cur_shard_id = n_done // SHARD_SIZE
    if n_done % SHARD_SIZE != 0:
        # Existing partial shard at cur_shard_id; we'll overwrite it on next flush.
        # (The latents shard for this range is incomplete; safest to just rewrite.)
        pass

    pending = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_thread) as ex:
        idx = n_done
        while idx < num_samples:
            this_b = min(batch_size, num_samples - idx)
            z = torch.randn(this_b, *z_shape, device=device)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                if sampler == "ddim":
                    out = diffusion.ddim_sample_loop(
                        model.forward, z.shape, z,
                        clip_denoised=False, model_kwargs={}, progress=False,
                        device=device, eta=ddim_eta,
                    )
                else:
                    out = diffusion.p_sample_loop(
                        model.forward, z.shape, z,
                        clip_denoised=False, model_kwargs={}, progress=False, device=device,
                    )
            imgs = decode(out)              # (B, 3, H, W) float in ~[0,1]
            # Wait for previous batch's PNG saves before queuing more
            for f in pending: f.result()
            pending = save_pngs_async(imgs, start_idx=idx, images_dir=images_dir, executor=ex)

            shard_buf_lat.append(out.float().cpu())
            shard_buf_idx.extend(range(idx, idx + this_b))
            idx += this_b

            # Flush a shard whenever its boundary is crossed
            while sum(t.shape[0] for t in shard_buf_lat) >= SHARD_SIZE:
                B_total = sum(t.shape[0] for t in shard_buf_lat)
                concat = torch.cat(shard_buf_lat, dim=0)
                take = SHARD_SIZE
                shard_lat = concat[:take]
                shard_idx = shard_buf_idx[:take]
                shard_path = os.path.join(latents_dir, f"shard_{cur_shard_id:04d}.pt")
                torch.save({"latents": shard_lat, "indices": shard_idx}, shard_path)
                cur_shard_id += 1
                # remainder for next shard
                rem_lat = concat[take:]
                rem_idx = shard_buf_idx[take:]
                shard_buf_lat = [rem_lat] if rem_lat.shape[0] > 0 else []
                shard_buf_idx = rem_idx

            elapsed = time.time() - t0
            done = idx - n_done
            rate = done / max(1e-6, elapsed)
            eta  = (num_samples - idx) / max(1e-6, rate)
            print(f"[gen] {idx}/{num_samples} done; rate={rate:.2f} img/s; ETA={eta/60:.1f} min",
                  flush=True)

        for f in pending: f.result()

    # Flush trailing partial shard
    if sum(t.shape[0] for t in shard_buf_lat) > 0:
        concat = torch.cat(shard_buf_lat, dim=0)
        shard_path = os.path.join(latents_dir, f"shard_{cur_shard_id:04d}.pt")
        torch.save({"latents": concat, "indices": shard_buf_idx}, shard_path)

    print(f"[DONE] total samples in {images_dir}: {count_existing_pngs(images_dir)}")
    print(f"[DONE] latent shards in {latents_dir}: "
          f"{len(glob.glob(os.path.join(latents_dir, 'shard_*.pt')))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, required=True,
                   choices=["our_method", "shared_bases", "no_alignment", "data_augmentation"])
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    p.add_argument("--num-sampling-steps", type=int, default=250)
    p.add_argument("--ddim-eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", type=str, default=None,
                   help="explicit checkpoint path; if omitted, uses find_latest_ckpt()")
    p.add_argument("--samples-subdir", type=str, default="samples",
                   help="subdirectory under {results}/{method}/ to write images/latents")
    a = p.parse_args()
    run(method=a.method, num_samples=a.num_samples, batch_size=a.batch_size,
        sampler=a.sampler, num_sampling_steps=a.num_sampling_steps,
        ddim_eta=a.ddim_eta, seed=a.seed, ckpt=a.ckpt, samples_subdir=a.samples_subdir)
