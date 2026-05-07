#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast
sys.path.insert(0, os.path.dirname(__file__))
from dit_models import JointDiT
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from diffusion import create_diffusion


# Checkpoint helpers

def find_latest_checkpoint(results_dir: str) -> str:
    numbered = sorted(
        glob.glob(os.path.join(results_dir, "*", "checkpoints", "[0-9]" * 7 + ".pt"))
    )
    if numbered:
        return numbered[-1]
    finals = sorted(
        glob.glob(os.path.join(results_dir, "*", "checkpoints", "final.pt"))
    )
    if finals:
        return finals[-1]
    raise FileNotFoundError(f"No checkpoints found under {results_dir}/*/checkpoints/")


def load_joint_dit(
    ckpt_path, use_ema, device,
    hidden_size, depth, num_heads, mlp_ratio,
    img_hw, patch, svd_rank,
):
    model = JointDiT(
        hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
        img_size=img_hw, patch_size=patch, rank=svd_rank,
    ).to(device)

    def _strip_prefix(k):
        for p in ("_orig_mod.", "module."):
            if k.startswith(p):
                return k[len(p):]
        return k

    try:
        torch.serialization.add_safe_globals([argparse.Namespace])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt):
        sd = ckpt["ema"] if (use_ema and ckpt.get("ema") is not None) else ckpt["model"]
        print(f"[INFO] Using {'EMA' if use_ema else 'raw model'} weights")
    else:
        sd = ckpt

    sd = {_strip_prefix(k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


# Resume helpers

_PNG_RE = re.compile(r"^(\d{5})\.png$")

def scan_existing_pngs(output_dir: str):
    if not os.path.isdir(output_dir):
        return 0, []
    indexed_files = []
    ignored_files = []
    for name in os.listdir(output_dir):
        full_path = os.path.join(output_dir, name)
        if not os.path.isfile(full_path):
            continue
        m = _PNG_RE.match(name)
        if m:
            indexed_files.append((int(m.group(1)), name))
        else:
            ignored_files.append(name)
    indexed_files.sort(key=lambda x: x[0])
    if ignored_files:
        print(f"[INFO] Ignoring non-matching files: {ignored_files[:5]}")
    for expected_idx, (actual_idx, name) in enumerate(indexed_files):
        if actual_idx != expected_idx:
            raise RuntimeError(
                f"PNG files not contiguous from 00000.png. "
                f"Expected {expected_idx:05d}, found {name}."
            )
    return len(indexed_files), [name for _, name in indexed_files]


# Async PNG saving with thread pool

def _save_one_png(args_tuple):
    """Worker: convert float tensor -> PIL -> PNG, called in thread pool."""
    tensor_hwc_uint8, path = args_tuple
    img = Image.fromarray(tensor_hwc_uint8)
    img.save(path, format="PNG", compress_level=1)   # compress_level=1: fastest PNG


def save_images_async(
    images: torch.Tensor,          # (B, C, H, W) float [0,1] on CPU
    start_idx: int,
    output_dir: str,
    executor: ThreadPoolExecutor,
) -> list:
    """
    Submits PNG saves to a thread pool.
    Returns list of futures -- call future.result() before next batch
    to avoid unbounded memory growth.
    
    Key changes vs original:
      - batch-convert to uint8 on CPU (one fast tensor op)
      - compress_level=1 (much faster than default=6)
      - parallel saves across threads
    """
    # (B, C, H, W) float -> uint8 -> (B, H, W, C) numpy -- one vectorised op
    imgs_uint8 = (images * 255).clamp(0, 255).byte()       
    imgs_np = imgs_uint8.permute(0, 2, 3, 1).numpy()       # (B, H, W, C)

    futures = []
    for i in range(imgs_np.shape[0]):
        path = os.path.join(output_dir, f"{start_idx + i:05d}.png")
        fut = executor.submit(_save_one_png, (imgs_np[i], path))
        futures.append(fut)
    return futures


# Decode: joint latent -> RGB image

@torch.no_grad()
def decode_joint(
    joint: torch.Tensor,
    alpha_rank_std: Optional[torch.Tensor],
    norm_std: float,
    vhat_std: Optional[torch.Tensor],
    mean: torch.Tensor,
    patch: int = 8,
    img_hw: int = 1024,
) -> torch.Tensor:
    device = joint.device
    B, C, H_joint, R = joint.shape

    patch_dim = patch * patch
    tokens_per_side = img_hw // patch
    alpha_tokens = tokens_per_side * tokens_per_side
    expected_joint_h = alpha_tokens + patch_dim

    if H_joint != expected_joint_h:
        raise ValueError(
            f"joint.shape[2]={H_joint}, expected {expected_joint_h}"
        )

    alpha = joint[:, :, :alpha_tokens, :].float()
    V_hat = joint[:, :, alpha_tokens:, :].float()

    if alpha_rank_std is not None:
        std_dev = alpha_rank_std.to(device)[None, :, None, :]
        alpha = alpha * std_dev
    else:
        alpha = alpha * norm_std

    if vhat_std is not None:
        V_hat = V_hat * vhat_std.to(device)

    A_hat = torch.bmm(
        alpha.reshape(B * C, alpha_tokens, R),
        V_hat.reshape(B * C, patch_dim, R).transpose(1, 2),
    )

    mean_ = mean.to(device)
    A_hat = A_hat.reshape(B, C, alpha_tokens, patch_dim) + mean_[None, :, None, :]

    nh = nw = tokens_per_side
    x_hat = (
        A_hat.reshape(B, C, nh, nw, patch, patch)
             .permute(0, 1, 2, 4, 3, 5)
             .contiguous()
             .reshape(B, C, img_hw, img_hw)
    )

    return x_hat.clamp(0.0, 1.0)


# Sampling

@torch.no_grad()
def run_sampling(diffusion, model, z, device, sampler="ddim", ddim_eta=0.0):
    common_kwargs = dict(
        model=model.forward,
        shape=z.shape,
        noise=z,
        clip_denoised=False,
        model_kwargs={},
        progress=False,
        device=device,
    )
    if sampler == "ddim":
        if not hasattr(diffusion, "ddim_sample_loop"):
            raise AttributeError("diffusion object does not implement ddim_sample_loop().")
        return diffusion.ddim_sample_loop(**common_kwargs, eta=ddim_eta)
    if sampler == "ddpm":
        return diffusion.p_sample_loop(**common_kwargs)
    raise ValueError(f"Unknown sampler: {sampler}")


# Main

def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.img_hw % args.patch != 0:
        raise ValueError(f"img_hw={args.img_hw} must be divisible by patch={args.patch}")

    tokens_per_side = args.img_hw // args.patch
    alpha_tokens = tokens_per_side * tokens_per_side
    patch_dim = args.patch * args.patch
    joint_h = alpha_tokens + patch_dim
    joint_w = args.svd_rank

    print("[INFO] Derived shapes:")
    print(f"       img_hw       = {args.img_hw}")
    print(f"       patch        = {args.patch}")
    print(f"       svd_rank     = {args.svd_rank}")
    print(f"       alpha_tokens = {alpha_tokens}")
    print(f"       patch_dim    = {patch_dim}")
    print(f"       joint shape  = (B, 3, {joint_h}, {joint_w})")

    # Load 
    ckpt_path = args.ckpt if args.ckpt else find_latest_checkpoint(args.results_dir)
    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    alpha_rank_std = None
    if args.alpha_stats_path and os.path.exists(args.alpha_stats_path):
        stats = torch.load(args.alpha_stats_path, map_location="cpu", weights_only=False)
        alpha_rank_std = stats["std"].float()
        print(f"[INFO] alpha_rank_std shape={tuple(alpha_rank_std.shape)}")
    else:
        print(f"[INFO] No alpha stats; using scalar norm_std={args.norm_std}")

    vhat_std = None
    if args.vhat_stats_path and os.path.exists(args.vhat_stats_path):
        vckpt = torch.load(args.vhat_stats_path, map_location="cpu", weights_only=False)
        vhat_std = vckpt["std"].float()
        print(f"[INFO] vhat_std = {vhat_std.item() if vhat_std.numel()==1 else tuple(vhat_std.shape)}")
    else:
        print("[INFO] No vhat stats; V_hat will not be de-normalised.")

    mean = torch.zeros(3, patch_dim)
    if args.ref_anchor_path and os.path.exists(args.ref_anchor_path):
        anchor = torch.load(args.ref_anchor_path, map_location="cpu", weights_only=False)
        mean = anchor["mean_ref"].float()
        print(f"[INFO] Loaded mean_ref from ref_anchor, shape={tuple(mean.shape)}")
    elif args.dict_path and os.path.exists(args.dict_path):
        dckpt = torch.load(args.dict_path, map_location="cpu", weights_only=False)
        if "mean" in dckpt:
            mean = dckpt["mean"].float()
    else:
        print("[INFO] Using zero mean.")

    # Model & diffusion 
    model = load_joint_dit(
        ckpt_path=ckpt_path, use_ema=args.use_ema, device=device,
        hidden_size=args.hidden_size, depth=args.depth,
        num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
        img_hw=args.img_hw, patch=args.patch, svd_rank=args.svd_rank,
    )

    # Compile model for ~20-30% speedup
    if args.compile:
        print("[INFO] Compiling model with torch.compile ...")
        model = torch.compile(model)

    timestep_respacing = (
        f"ddim{args.num_sampling_steps}"
        if args.sampler == "ddim"
        else str(args.num_sampling_steps)
    )
    diffusion = create_diffusion(
        timestep_respacing,
        learn_sigma=False,
        predict_xstart=True,
    )

    print(f"[INFO] Sampler        = {args.sampler}")
    print(f"[INFO] Sampling steps = {args.num_sampling_steps}")
    if args.sampler == "ddim":
        print(f"[INFO] DDIM eta       = {args.ddim_eta}")

    # Resume 
    os.makedirs(args.output_dir, exist_ok=True)
    existing_count, _ = scan_existing_pngs(args.output_dir)
    print(f"[INFO] Output dir   : {args.output_dir}")
    print(f"[INFO] Existing PNGs: {existing_count}")

    if existing_count >= args.num_images:
        print(f"[INFO] Already have {existing_count} images. Nothing to do.")
        return

    remaining = args.num_images - existing_count
    print(f"[INFO] Need {remaining} more images.")

    # Thread pool for async PNG saving
    # num_save_threads: rule of thumb = min(8, cpu_count)
    # each thread compresses one 1024x1024 PNG indep.
    num_save_threads = min(8, os.cpu_count() or 4)
    print(f"[INFO] PNG save threads = {num_save_threads}  (compress_level=1)")

    generated = existing_count
    total_batches = (remaining + args.batch_size - 1) // args.batch_size
    batch_idx = 0
    pending_futures = []          # outstanding save futures from prev batch

    pbar = tqdm(total=args.num_images, initial=generated, desc="Generating", unit="img")

    with ThreadPoolExecutor(max_workers=num_save_threads) as executor:
        while generated < args.num_images:
            batch = min(args.batch_size, args.num_images - generated)
            batch_idx += 1
            pbar.set_description(
                f"Batch {batch_idx}/{total_batches} (done={generated}, size={batch})"
            )

            # Wait for previous batch's saves before generating next
            # (avoids RAM explosion; saves overlap with GPU work)
            for fut in pending_futures:
                fut.result()       
            pending_futures = []

            # GPU: sample 
            # FIX 3: drop channels_last (wrong for tall latents)
            z = torch.randn(batch, 3, joint_h, joint_w, device=device)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                samples = run_sampling(
                    diffusion=diffusion, model=model, z=z, device=device,
                    sampler=args.sampler, ddim_eta=args.ddim_eta,
                )

            # CPU: decode 
            images = decode_joint(
                samples.float(),
                alpha_rank_std=alpha_rank_std,
                norm_std=args.norm_std,
                vhat_std=vhat_std,
                mean=mean,
                patch=args.patch,
                img_hw=args.img_hw,
            )

            # Async save 
            images_cpu = images.cpu()
            pending_futures = save_images_async(
                images_cpu, start_idx=generated,
                output_dir=args.output_dir, executor=executor,
            )
            generated += batch
            pbar.update(batch)

        # Flush last batch
        for fut in pending_futures:
            fut.result()

    pbar.close()
    print(f"\nDone. {generated} images now present in '{args.output_dir}'.")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Output dir : {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from Procrustes JointDiT for FID evaluation (fast version)"
    )

    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--depth",       type=int, default=12)
    parser.add_argument("--num-heads",   type=int, default=12)
    parser.add_argument("--mlp-ratio",   type=float, default=4.0)

    parser.add_argument("--model-patch-h", type=int, default=32)
    parser.add_argument("--model-patch-w", type=int, default=32)

    parser.add_argument("--ckpt",        type=str, default=None)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--use-ema",     action="store_true")

    parser.add_argument("--alpha-stats-path", type=str, default=None)
    parser.add_argument("--vhat-stats-path",  type=str, default=None)
    parser.add_argument("--dict-path",        type=str, default=None)
    parser.add_argument("--ref-anchor-path",  type=str, default=None)
    parser.add_argument("--norm-std",         type=float, default=0.5)

    parser.add_argument("--img-hw",   type=int, default=1024)
    parser.add_argument("--patch",    type=int, default=32)
    parser.add_argument("--svd-rank", type=int, default=16)

    parser.add_argument("--output-dir",  type=str, required=True)
    parser.add_argument("--num-images",  type=int, default=5000)
    parser.add_argument("--batch-size",  type=int, default=128)

    parser.add_argument(
        "--sampler", type=str, default="ddim", choices=["ddpm", "ddim"],
    )
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--ddim-eta",           type=float, default=0.0)
    parser.add_argument("--seed",               type=int, default=42)

    # Optional: torch.compile flag
    parser.add_argument(
        "--compile", action="store_true",
        help="Use torch.compile() for ~20-30%% extra GPU speedup (PyTorch 2.x only)."
    )

    args = parser.parse_args()
    main(args)
