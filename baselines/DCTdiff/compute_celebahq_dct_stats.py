#!/usr/bin/env python3
"""
Compute RGB DCT statistics for the direct-RGB DCTdiff pipeline.

This script:
1. Extracts block DCT coefficients for the R/G/B channels into sharded arrays.
2. Computes a shared scalar coefficient bound (`Y_bound`) for normalization.
3. Computes per-channel entropy-style frequency weights (`R_std`, `G_std`, `B_std`).
4. Writes the result to both:
   - `<output_dir>/<dataset>_<block>by<block>_rgb_stats.json`
   - `<output_dir>/<dataset>_<block>by<block>_rgb_stats.txt`

Example:
  python compute_rgb_dct_stats.py \
    --dataset celebahq1024 \
    --img-folder ../CelebA-HQ/celeba_hq_images/all \
    --block-sz 16 \
    --keep-coeffs 16 \
    --step all
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
from PIL import Image
from tqdm import tqdm

from DCT_utils import dct_transform, split_into_blocks, zigzag_order

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")
CHANNELS = ("r", "g", "b")


def _shard_index(path: str) -> int:
    match = re.search(r"_(\d+)\.npy$", path)
    return int(match.group(1)) if match else 0


def _iter_image_paths(root: str):
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename.lower().endswith(IMAGE_EXTS):
                yield os.path.join(dirpath, filename)


def _stats_prefix(output_dir: str, dataset: str, block_sz: int) -> str:
    return os.path.join(output_dir, f"{dataset}_{block_sz}by{block_sz}")


def _channel_shard_paths(prefix: str, channel: str) -> list[str]:
    paths = sorted(glob.glob(f"{prefix}_{channel}_*.npy"), key=_shard_index)
    if not paths:
        raise FileNotFoundError(
            f"No shards found for channel '{channel}' with prefix {prefix}. "
            "Run extraction first with --step extract or --step all."
        )
    return paths


def _flush_channel_buffers(prefix: str, buffers: dict[str, list[np.ndarray]], shard_idx: int) -> None:
    for channel in CHANNELS:
        if not buffers[channel]:
            continue
        array = np.concatenate(buffers[channel], axis=0).astype(np.float16, copy=False)
        out_path = f"{prefix}_{channel}_{shard_idx}.npy"
        np.save(out_path, array)
        print(f"[save] {out_path} {array.shape}")
        buffers[channel].clear()


def extract_rgb_dct_shards(
    dataset: str,
    img_folder: str,
    block_sz: int,
    output_dir: str,
    shard_images: int,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    prefix = _stats_prefix(output_dir, dataset, block_sz)
    image_paths = list(_iter_image_paths(img_folder))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {img_folder}")

    print(f"[extract] found {len(image_paths)} images in {img_folder}")
    buffers = {channel: [] for channel in CHANNELS}
    shard_idx = 1
    images_in_shard = 0

    for img_path in tqdm(image_paths, desc="extract_rgb_dct"):
        img = np.array(Image.open(img_path).convert("RGB"))
        for channel_name, channel_idx in zip(CHANNELS, range(3)):
            blocks = split_into_blocks(img[:, :, channel_idx], block_sz)
            dct_blocks = dct_transform(blocks).reshape(-1, block_sz * block_sz)
            buffers[channel_name].append(dct_blocks)

        images_in_shard += 1
        if images_in_shard >= shard_images:
            _flush_channel_buffers(prefix, buffers, shard_idx)
            shard_idx += 1
            images_in_shard = 0

    if images_in_shard:
        _flush_channel_buffers(prefix, buffers, shard_idx)

    return prefix


def _load_column(prefix: str, channel: str, coeff_idx: int) -> np.ndarray:
    parts = []
    for path in _channel_shard_paths(prefix, channel):
        shard = np.load(path, mmap_mode="r")
        parts.append(np.asarray(shard[:, coeff_idx], dtype=np.float32))
    return np.concatenate(parts, axis=0)


def _load_columns_chunk(shard_paths: list[str], coeff_indices: list[int]) -> np.ndarray:
    parts = []
    for path in shard_paths:
        shard = np.load(path, mmap_mode="r")
        parts.append(np.asarray(shard[:, coeff_indices], dtype=np.float32))
    return np.concatenate(parts, axis=0)


def _trim_by_percentile(values: np.ndarray, tau: float) -> tuple[np.ndarray, float, float]:
    low_thresh = 100.0 - tau
    lower = float(np.percentile(values, low_thresh))
    upper = float(np.percentile(values, tau))
    filtered = values[(values >= lower) & (values <= upper)]
    if filtered.size == 0:
        filtered = values
    return filtered, lower, upper


def _selected_coeff_indices(block_sz: int, keep_coeffs: int | None) -> list[int]:
    total_coeffs = block_sz * block_sz
    if keep_coeffs is None:
        return list(range(total_coeffs))
    if keep_coeffs <= 0 or keep_coeffs > total_coeffs:
        raise ValueError(f"keep_coeffs must be in [1, {total_coeffs}], got {keep_coeffs}")
    return list(zigzag_order(block_sz)[:keep_coeffs])


def compute_shared_bound(
    prefix: str,
    block_sz: int,
    tau: float,
    mode: str,
    coeff_chunk_size: int,
    keep_coeffs: int | None,
) -> float:
    coeff_indices = [0] if mode == "dc" else _selected_coeff_indices(block_sz, keep_coeffs)
    bound = 0.0
    total_coeffs = len(CHANNELS) * len(coeff_indices)
    with tqdm(total=total_coeffs, desc="compute_bound") as pbar:
        for channel in CHANNELS:
            shard_paths = _channel_shard_paths(prefix, channel)
            for chunk_start in range(0, len(coeff_indices), coeff_chunk_size):
                chunk_coeffs = coeff_indices[chunk_start:chunk_start + coeff_chunk_size]
                chunk_values = _load_columns_chunk(shard_paths, chunk_coeffs)
                for local_idx, coeff_idx in enumerate(chunk_coeffs):
                    values = chunk_values[:, local_idx]
                    _, lower, upper = _trim_by_percentile(values, tau)
                    bound = max(bound, abs(lower), abs(upper))
                    print(
                        f"[bound] channel={channel.upper()} coeff={coeff_idx} "
                        f"lower={lower:.3f} upper={upper:.3f} running_bound={bound:.3f}"
                    )
                    pbar.update(1)
    bound = round(bound, 3)
    print(f"[bound] shared coefficient bound = {bound}")
    return bound


def compute_entropy_weights(
    prefix: str,
    channel: str,
    block_sz: int,
    tau: float,
    bound: float,
    coeff_chunk_size: int,
    keep_coeffs: int | None,
) -> list[float]:
    entropies = np.ones(block_sz * block_sz, dtype=np.float32)
    shard_paths = _channel_shard_paths(prefix, channel)
    selected_coeffs = _selected_coeff_indices(block_sz, keep_coeffs)
    with tqdm(total=len(selected_coeffs), desc=f"entropy_{channel}") as pbar:
        for chunk_start in range(0, len(selected_coeffs), coeff_chunk_size):
            chunk_coeffs = selected_coeffs[chunk_start:chunk_start + coeff_chunk_size]
            chunk_values = _load_columns_chunk(shard_paths, chunk_coeffs)
            for local_idx, coeff_idx in enumerate(chunk_coeffs):
                values = chunk_values[:, local_idx]
                filtered, lower, upper = _trim_by_percentile(values, tau)
                normalized = filtered / bound
                counts, _ = np.histogram(normalized, bins=100, range=(-1, 1))
                total = max(int(np.sum(counts)), 1)
                probabilities = counts / total
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
                entropy = round(float(entropy), 3)
                entropies[coeff_idx] = entropy
                print(
                    f"[entropy] channel={channel.upper()} coeff={coeff_idx} "
                    f"lower={lower:.3f} upper={upper:.3f} entropy={entropy:.3f}"
                )
                pbar.update(1)
    return entropies.tolist()


def compute_per_freq_bounds(
    img_folder: str,
    block_sz: int,
    tau: float = 99.0,
    n_sample: int = 5000,
) -> list[float]:
    """Per-frequency DCT coefficient bounds.

    For each of block_sz**2 spatial coefficient positions, takes the maximum of
    the per-image per-channel tau-th percentile of |coeff| across a sample of
    images.  Peak working memory is ~4 MB per image regardless of dataset size.
    """
    import random

    n_coeffs = block_sz * block_sz
    image_paths = list(_iter_image_paths(img_folder))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {img_folder}")
    if len(image_paths) > n_sample:
        random.seed(42)
        image_paths = random.sample(image_paths, n_sample)
    print(f"[per_freq_bounds] {len(image_paths)} images, block_sz={block_sz}, tau={tau}")

    bounds = np.zeros(n_coeffs, dtype=np.float64)
    for img_path in tqdm(image_paths, desc="per_freq_bounds"):
        img = np.array(Image.open(img_path).convert("RGB"))
        for channel_idx in range(3):
            blocks = split_into_blocks(img[:, :, channel_idx], block_sz)
            abs_dct = np.abs(dct_transform(blocks)).reshape(-1, n_coeffs)  # (n_blocks, n_coeffs)
            img_bounds = np.percentile(abs_dct, tau, axis=0)              # (n_coeffs,)
            bounds = np.maximum(bounds, img_bounds)

    bounds = np.maximum(bounds, 1.0)  # avoid div/0 for zero-energy high-freq coefficients
    print(f"[per_freq_bounds] DC={bounds[0]:.1f}  coeff_1={bounds[1]:.1f}  min={bounds.min():.2f}")
    return bounds.tolist()


def _stats_payload(
    y_bound: float,
    r_std: list[float],
    g_std: list[float],
    b_std: list[float],
    y_bound_per_freq: list[float] | None = None,
) -> dict:
    payload = {
        "Y_bound": [y_bound],
        "R_std": r_std,
        "G_std": g_std,
        "B_std": b_std,
    }
    if y_bound_per_freq is not None:
        payload["Y_bound_per_freq"] = y_bound_per_freq
    return payload


def write_stats_json(
    prefix: str,
    y_bound: float,
    r_std: list[float],
    g_std: list[float],
    b_std: list[float],
    y_bound_per_freq: list[float] | None = None,
) -> str:
    out_path = f"{prefix}_rgb_stats.json"
    payload = _stats_payload(y_bound, r_std, g_std, b_std, y_bound_per_freq)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"[save] {out_path}")
    return out_path


def write_stats_txt(
    prefix: str,
    y_bound: float,
    r_std: list[float],
    g_std: list[float],
    b_std: list[float],
    y_bound_per_freq: list[float] | None = None,
) -> str:
    out_path = f"{prefix}_rgb_stats.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Y_bound={[y_bound]}\n")
        f.write(f"R_std={r_std}\n")
        f.write(f"G_std={g_std}\n")
        f.write(f"B_std={b_std}\n")
        if y_bound_per_freq is not None:
            f.write(f"Y_bound_per_freq={y_bound_per_freq}\n")
    print(f"[save] {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RGB DCT stats for direct-RGB DCTdiff.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name prefix for saved shard files")
    parser.add_argument("--img-folder", type=str, required=True, help="Folder containing RGB images")
    parser.add_argument("--block-sz", type=int, required=True, help="DCT block size")
    parser.add_argument(
        "--keep-coeffs",
        type=int,
        default=None,
        help=(
            "If set, compute stats only for the first N zigzag coefficients used by training. "
            "Saved arrays still have length block_sz**2 so existing training code works unchanged."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dct_arrays"),
        help="Directory for DCT shard files and final JSON stats",
    )
    parser.add_argument(
        "--step",
        choices=("extract", "stats", "all"),
        default="all",
        help="Whether to extract shards, compute stats from existing shards, or do both",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=98.25,
        help="Percentile threshold used for trimmed bound/entropy statistics",
    )
    parser.add_argument(
        "--shard-images",
        type=int,
        default=16,
        help="Number of images buffered per shard during extraction",
    )
    parser.add_argument(
        "--bound-mode",
        choices=("dc", "max-all"),
        default="dc",
        help="Use only the DC coefficient for the shared bound, or scan all coefficients",
    )
    parser.add_argument(
        "--coeff-chunk-size",
        type=int,
        default=8,
        help="Number of coefficients to load per shard pass during stats. Larger is faster but uses more RAM.",
    )
    parser.add_argument(
        "--per-freq-tau",
        type=float,
        default=99.0,
        help="Percentile used for per-frequency bounds (higher = larger bounds = safer but less normalised).",
    )
    parser.add_argument(
        "--per-freq-n-sample",
        type=int,
        default=5000,
        help="Max images to scan when computing per-frequency bounds.",
    )
    args = parser.parse_args()

    prefix = _stats_prefix(args.output_dir, args.dataset, args.block_sz)
    if args.step in ("extract", "all"):
        prefix = extract_rgb_dct_shards(
            dataset=args.dataset,
            img_folder=args.img_folder,
            block_sz=args.block_sz,
            output_dir=args.output_dir,
            shard_images=args.shard_images,
        )

    if args.step in ("stats", "all"):
        bound = compute_shared_bound(
            prefix, args.block_sz, args.tau, args.bound_mode, args.coeff_chunk_size, args.keep_coeffs
        )
        r_std = compute_entropy_weights(
            prefix, "r", args.block_sz, args.tau, bound, args.coeff_chunk_size, args.keep_coeffs
        )
        g_std = compute_entropy_weights(
            prefix, "g", args.block_sz, args.tau, bound, args.coeff_chunk_size, args.keep_coeffs
        )
        b_std = compute_entropy_weights(
            prefix, "b", args.block_sz, args.tau, bound, args.coeff_chunk_size, args.keep_coeffs
        )
        y_bound_per_freq = compute_per_freq_bounds(
            args.img_folder, args.block_sz, tau=args.per_freq_tau, n_sample=args.per_freq_n_sample
        )
        payload = _stats_payload(bound, r_std, g_std, b_std, y_bound_per_freq)
        write_stats_json(prefix, bound, r_std, g_std, b_std, y_bound_per_freq)
        write_stats_txt(prefix, bound, r_std, g_std, b_std, y_bound_per_freq)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
