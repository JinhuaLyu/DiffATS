"""
Generate a slow-motion Moving MNIST by re-simulating digit trajectories on GPU.

Physics simulation (trajectory) and digit rendering are fully vectorized on GPU.
No interpolation -- digits are crisp pixel-level renders at each frame.

Output: slow_moving_mnist.pt  --  torch.Tensor shape (20, N, 64, 64) uint8

Usage:
    conda run -n rpy2-env python3 generate_slow_moving_mnist.py
    conda run -n rpy2-env python3 generate_slow_moving_mnist.py --n_videos 200 --preview
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

CANVAS = 64
DIGIT_SIZE = 28
N_FRAMES = 20
N_DIGITS = 2
SPEED_FACTOR = 0.5              # 2x slowdown
ORIG_SPEED_RANGE = (2.0, 4.0)  # px/frame magnitude range of original Moving MNIST
MAX_POS = CANVAS - DIGIT_SIZE  # 36
MIN_INIT_DIST = 28              # minimum Euclidean distance between digit top-left corners

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MNIST_CACHE = os.path.join(OUT_DIR, ".mnist_cache")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mnist_test_images() -> torch.Tensor:
    """Return MNIST test images as (10000, 28, 28) float32."""
    import torchvision
    dataset = torchvision.datasets.MNIST(root=MNIST_CACHE, train=False, download=True)
    return dataset.data.float()   # (10000, 28, 28)


# ---------------------------------------------------------------------------
# GPU-vectorized batch generation
# ---------------------------------------------------------------------------

def generate_batch(digit_imgs: torch.Tensor, rng: np.random.Generator,
                   device: torch.device) -> torch.Tensor:
    """
    Generate one batch of videos entirely on GPU.

    digit_imgs : (B, N_DIGITS, 28, 28) float32 [0-255], on GPU
    Returns    : (B, N_FRAMES, 64, 64) uint8,   on GPU
    """
    B = digit_imgs.shape[0]

    # Random initial positions with minimum distance constraint (rejection sampling)
    pos_np = rng.uniform(0, MAX_POS, size=(B, N_DIGITS, 2))
    while True:
        diff = pos_np[:, 0, :] - pos_np[:, 1, :]   # (B, 2)
        dist = np.linalg.norm(diff, axis=-1)         # (B,)
        bad = dist < MIN_INIT_DIST
        if not bad.any():
            break
        pos_np[bad] = rng.uniform(0, MAX_POS, size=(int(bad.sum()), N_DIGITS, 2))

    pos = torch.tensor(pos_np, dtype=torch.float32, device=device)
    # (B, N_DIGITS, 2)  -- dim 2: [x, y]

    angles = torch.tensor(
        rng.uniform(0, 2 * np.pi, size=(B, N_DIGITS)),
        dtype=torch.float32, device=device,
    )
    speeds = torch.tensor(
        rng.uniform(*ORIG_SPEED_RANGE, size=(B, N_DIGITS)) * SPEED_FACTOR,
        dtype=torch.float32, device=device,
    )
    vel = torch.stack([speeds * torch.cos(angles),
                       speeds * torch.sin(angles)], dim=-1)  # (B, N_DIGITS, 2)

    # Pixel offset grids for a 28x28 digit patch
    dr = torch.arange(DIGIT_SIZE, device=device)
    dc = torch.arange(DIGIT_SIZE, device=device)
    grid_r, grid_c = torch.meshgrid(dr, dc, indexing="ij")  # (28, 28)

    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, DIGIT_SIZE, DIGIT_SIZE)

    all_frames = []

    for _ in range(N_FRAMES):
        canvas = torch.zeros(B, CANVAS, CANVAS, dtype=torch.float32, device=device)

        pos_int = pos.long().clamp(0, MAX_POS)  # (B, N_DIGITS, 2)

        for d in range(N_DIGITS):
            # row/col indices for every pixel of every video in this digit slot
            row_idx = pos_int[:, d, 1].view(B, 1, 1) + grid_r  # (B, 28, 28)
            col_idx = pos_int[:, d, 0].view(B, 1, 1) + grid_c  # (B, 28, 28)

            b_flat   = b_idx.reshape(-1)
            row_flat = row_idx.reshape(-1)
            col_flat = col_idx.reshape(-1)
            dig_flat = digit_imgs[:, d].reshape(-1)  # (B*28*28,)

            existing = canvas[b_flat, row_flat, col_flat]
            canvas[b_flat, row_flat, col_flat] = torch.maximum(existing, dig_flat)

        all_frames.append(canvas)

        # Physics: update positions and bounce off walls
        pos = pos + vel

        for dim in range(2):
            lo = pos[:, :, dim] < 0
            hi = pos[:, :, dim] > MAX_POS
            pos[:, :, dim] = torch.where(lo, -pos[:, :, dim], pos[:, :, dim])
            pos[:, :, dim] = torch.where(hi, 2.0 * MAX_POS - pos[:, :, dim], pos[:, :, dim])
            vel[:, :, dim] = torch.where(lo,  vel[:, :, dim].abs(),  vel[:, :, dim])
            vel[:, :, dim] = torch.where(hi, -vel[:, :, dim].abs(), vel[:, :, dim])

    # (B, N_FRAMES, 64, 64) uint8
    return torch.stack(all_frames, dim=1).clamp(0, 255).to(torch.uint8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_gif(frames, path: str, duration_ms: int = 100):
    """frames: (T, H, W) uint8 numpy or CPU tensor -> GIF"""
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    imgs = [Image.fromarray(f, mode="L") for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0, duration=duration_ms)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_videos", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default=os.path.join(OUT_DIR, "slow_moving_mnist.pt"))
    parser.add_argument("--preview", action="store_true",
                        help="Save 3 comparison GIF pairs then exit")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading MNIST test images...")
    mnist_imgs = load_mnist_test_images().to(device)  # (10000, 28, 28)
    n_mnist = len(mnist_imgs)

    rng = np.random.default_rng(args.seed)
    n = 3 if args.preview else args.n_videos

    all_videos = []  # list of (B, 20, 64, 64) uint8 CPU tensors
    processed = 0

    while processed < n:
        bs = min(args.batch_size, n - processed)

        digit_idx = rng.integers(0, n_mnist, size=(bs, N_DIGITS))
        digit_imgs = mnist_imgs[digit_idx]  # (bs, N_DIGITS, 28, 28)

        batch = generate_batch(digit_imgs, rng, device)  # (bs, 20, 64, 64) uint8
        all_videos.append(batch.cpu())
        processed += bs
        print(f"  {processed}/{n}")

    result = torch.cat(all_videos, dim=0)  # (N, 20, 64, 64) uint8

    if args.preview:
        orig = torch.from_numpy(
            np.load("${HOME}/video_factor_diffusion/datasets/"
                    "moving_mnist/mnist_test_seq.npy")
        ).permute(1, 0, 2, 3)  # (10000, 20, 64, 64)

        for i in range(3):
            save_gif(orig[i], os.path.join(OUT_DIR, f"preview_original_{i}.gif"))
            save_gif(result[i], os.path.join(OUT_DIR, f"preview_slow2x_{i}.gif"))
            print(f"Saved preview pair {i}")
        print("Done. Compare preview_original_*.gif vs preview_slow2x_*.gif")
        return

    # Save as (20, N, 64, 64) to match original format
    out = result.permute(1, 0, 2, 3).contiguous()  # (20, N, 64, 64)
    torch.save(out, args.output)
    print(f"Saved: {args.output}  shape={tuple(out.shape)}  dtype={out.dtype}")


if __name__ == "__main__":
    main()
