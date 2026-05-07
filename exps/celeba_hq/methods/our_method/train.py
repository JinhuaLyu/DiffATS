# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
patch_size = 32 32
svd_rank = 32
patch_dim = 32*32 = 1024
alpha_n = (1024/32)^2 = 1024
joint_h = 1024 + 1024 = 2048
input_size = 2048 32
"""

import os
import sys
import io
import glob
import math
import argparse
import logging
import warnings
from time import time
from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
from collections import OrderedDict
from typing import Literal, Optional, Dict, Any, List, Tuple
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm
import wandb
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.amp import autocast

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.benchmark = True

try:
    torch.nn.attention.sdpa_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
except Exception:
    pass

sys.path.insert(0, os.path.dirname(__file__))
from dit_models import DiT, JointDiT
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from diffusion import create_diffusion

# -------------------------------------------------------------
# Data Augmentation
# -------------------------------------------------------------
def augmentation(data: torch.Tensor):
    """
    data: [B, C, H, W] where W == 16 (the SVD rank dimension)
      -> for each b in B, right-multiply a random Q_b in SO(16) on the rank dim
      -> output: same shape [B, C, H, W]
    """
    assert data.ndim == 4 and data.shape[-1] == 16, "data must be [B, C, H, 16]"
    B, C, H, W = data.shape
    device, dtype = data.device, data.dtype

    A = torch.randn(B, 16, 16, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A)

    detQ = torch.linalg.det(Q)
    neg = detQ < 0
    if neg.any():
        Q[neg, :, -1] *= -1

    y = torch.einsum("bchk,bkj->bchj", data, Q)
    return y


###########################################################
# Dataset Loading
###########################################################

class ShardedUVDataset(torch.utils.data.Dataset):
    """
    Dataset for shards saved as:
      {"U": (B, 256, r), "V": (B, 192, r), "filenames": [str, ...]}

    mode : {"uv","recon"}
        - "uv": returns (U, V, filename)
        - "recon": returns (A_hat, filename), where A_hat = U @ V^T
    """
    def __init__(
        self,
        shard_dir: str,
        mode: Literal["uv", "recon"] = "uv",
        preload: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*_shard_*.pt")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found in {shard_dir}")

        self.mode = mode
        self.preload = preload
        self.device = device

        self._meta: List[Tuple[int, int]] = []
        self._loaded_shards: Dict[int, Dict[str, Any]] = {}

        if self.preload:
            for i, p in enumerate(self.shard_paths):
                data = torch.load(p, map_location="cpu", weights_only=False)
                self._loaded_shards[i] = data
                b = data["U"].shape[0]
                for j in range(b):
                    self._meta.append((i, j))
        else:
            for i, p in enumerate(self.shard_paths):
                data = torch.load(p, map_location="cpu", weights_only=False)
                b = data["U"].shape[0]
                for j in range(b):
                    self._meta.append((i, j))
            self._current_shard: Optional[int] = None

    def __len__(self) -> int:
        return len(self._meta)

    def _ensure_shard_loaded(self, shard_idx: int):
        if self.preload:
            return
        if getattr(self, "_current_shard", None) != shard_idx:
            if getattr(self, "_current_shard", None) in self._loaded_shards:
                del self._loaded_shards[self._current_shard]
            path = self.shard_paths[shard_idx]
            self._loaded_shards[shard_idx] = torch.load(path, map_location="cpu", weights_only=False)
            self._current_shard = shard_idx

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._meta[idx]
        self._ensure_shard_loaded(shard_idx)
        data = self._loaded_shards[shard_idx]

        U = data["U"][local_idx]
        V = data["V"][local_idx]
        fname = data["filenames"][local_idx]

        if self.mode == "uv":
            if self.device is not None:
                U = U.to(self.device, non_blocking=True)
                V = V.to(self.device, non_blocking=True)
            return U, V, fname

        if self.mode == "recon":
            A_hat = U @ V.transpose(0, 1)
            if self.device is not None:
                A_hat = A_hat.to(self.device, non_blocking=True)
            return A_hat, fname

        raise ValueError(f"Unknown mode: {self.mode}")


class ShardedAlphaDataset(torch.utils.data.Dataset):
    """
    Dataset for global-dict projection shards saved as:
      {"alpha": (B, 3, N, r), "filenames": [str, ...], ...}

    Returns (alpha, filename) per sample.
    """
    def __init__(self, shard_dir: str, preload: bool = True):
        self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*_shard_*.pt")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No alpha shard files found in {shard_dir}")

        self.preload = preload
        self._meta: List[Tuple[int, int]] = []
        self._loaded_shards: Dict[int, Dict[str, Any]] = {}

        for i, p in enumerate(self.shard_paths):
            data = torch.load(p, map_location="cpu", weights_only=False)
            if self.preload:
                self._loaded_shards[i] = data
            b = data["alpha"].shape[0]
            for j in range(b):
                self._meta.append((i, j))

    def __len__(self) -> int:
        return len(self._meta)

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._meta[idx]
        if shard_idx not in self._loaded_shards:
            self._loaded_shards[shard_idx] = torch.load(
                self.shard_paths[shard_idx], map_location="cpu", weights_only=False
            )
        data = self._loaded_shards[shard_idx]
        alpha = data["alpha"][local_idx]
        fname = data["filenames"][local_idx]
        return alpha, fname


class ShardedProcAlphaDataset(torch.utils.data.Dataset):
    """
    Dataset for Procrustes-aligned per-image SVD shards saved as:
      {"alpha": (B, 3, N, r), "V_hat": (B, 3, d, r), "filenames": [...]}

    Returns (alpha, V_hat) per sample.
    """
    def __init__(self, shard_dir: str, preload: bool = True):
        self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*_shard_*.pt")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No procrustes shard files found in {shard_dir}")

        self.preload = preload
        self._meta: List[Tuple[int, int]] = []
        self._loaded_shards: Dict[int, Dict[str, Any]] = {}

        for i, p in enumerate(self.shard_paths):
            data = torch.load(p, map_location="cpu", weights_only=False)
            if self.preload:
                self._loaded_shards[i] = data
            b = data["alpha"].shape[0]
            for j in range(b):
                self._meta.append((i, j))

    def __len__(self) -> int:
        return len(self._meta)

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._meta[idx]
        if shard_idx not in self._loaded_shards:
            self._loaded_shards[shard_idx] = torch.load(
                self.shard_paths[shard_idx], map_location="cpu", weights_only=False
            )
        data = self._loaded_shards[shard_idx]
        alpha = data["alpha"][local_idx]
        V_hat = data["V_hat"][local_idx]
        return alpha, V_hat


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def requires_grad(model, flag: bool = True):
    for p in model.parameters():
        p.requires_grad = flag


def quiet_autotune_warmup(model, example_shape, device):
    B, C, H, W = example_shape
    x = torch.randn(B, C, H, W, device=device).contiguous(memory_format=torch.channels_last)
    t = torch.randint(0, 1000, (B,), device=device)

    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = model(x, t) if model.__class__.__name__.lower().startswith("unet") else model(x, t, None)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_name = name.removeprefix("_orig_mod.")
        if ema_name in ema_params:
            ema_params[ema_name].mul_(decay).add_(param.data, alpha=1 - decay)


def inv_normalize_svd(uprime: torch.Tensor, stats=None) -> torch.Tensor:
    m = stats["mean"]
    s = stats["std"]
    return uprime * s + m


def sample_S_from_stats(stats_path, num_samples=8, seed=1):
    torch.manual_seed(seed)
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    mu, cov, rank = stats["mu"], stats["cov"], stats["rank"]

    S_samples = []
    for _ in range(num_samples):
        logS_all = []
        for k in range(rank):
            mu_k = mu[k]
            cov_k = cov[k]
            L = torch.linalg.cholesky(cov_k)
            z = torch.randn(3)
            logS_k = mu_k + L @ z
            logS_all.append(logS_k.unsqueeze(-1))
        logS_all = torch.cat(logS_all, dim=-1)
        S = torch.exp(logS_all).clamp_min(1e-12)
        S_samples.append(S)
    return torch.stack(S_samples)


def infer_joint_layout(input_size, patch_size, svd_rank):
    """
    For JointDiT we interpret input_size=(joint_h, joint_w=rank).

    joint_h = alpha_n + patch_dim
    patch_dim = patch * patch
    alpha_n = (#patches in image) = grid_hw^2
    img_hw = grid_hw * patch
    """
    if isinstance(input_size, int):
        joint_h = int(input_size)
        joint_w = int(svd_rank)
    else:
        joint_h = int(input_size[0])
        joint_w = int(input_size[1])

    if isinstance(patch_size, int):
        ph = pw = int(patch_size)
    else:
        ph = int(patch_size[0])
        pw = int(patch_size[1])

    if ph != pw:
        raise ValueError(f"Only square patches supported for JointDiT here, got {patch_size}")

    patch = ph
    rank = int(svd_rank) if svd_rank > 0 else joint_w
    patch_dim = patch * patch
    alpha_n = joint_h - patch_dim

    if alpha_n <= 0:
        raise ValueError(
            f"Invalid joint layout: joint_h={joint_h}, patch_dim={patch_dim}, alpha_n={alpha_n}"
        )

    grid_hw = int(round(alpha_n ** 0.5))
    if grid_hw * grid_hw != alpha_n:
        raise ValueError(f"alpha_n={alpha_n} is not a perfect square.")

    img_hw = grid_hw * patch
    return {
        "img_hw": img_hw,
        "patch": patch,
        "rank": rank,
        "patch_dim": patch_dim,
        "alpha_n": alpha_n,
        "joint_h": joint_h,
        "joint_w": joint_w,
        "grid_hw": grid_hw,
    }


def recover_from_uv_to_image(
    U: torch.Tensor,
    V: torch.Tensor,
    patch: int = 8,
    img_hw: int = 128,
    clamp: bool = True,
):
    assert U.ndim == 3 and V.ndim == 3, "U, V must be (B,N,r) and (B,d,r)"
    B, N, r = U.shape
    B2, d, r2 = V.shape
    assert B == B2 and r == r2, "Batch size / rank mismatch"
    assert img_hw % patch == 0, "img_hw must be divisible by patch"
    nh = nw = img_hw // patch
    assert N == nh * nw, f"Expected N={nh*nw}, got {N}"
    assert d == patch * patch * 3, f"Expected d={patch*patch*3}, got {d}"

    A_hat = torch.bmm(U, V.transpose(1, 2))
    patches = A_hat.view(B, N, patch, patch, 3)
    patches = patches.view(B, nh, nw, patch, patch, 3)

    x_hat = (
        patches.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, img_hw, img_hw, 3)
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    if clamp:
        x_hat = x_hat.clamp(0.0, 1.0)
    return x_hat


def recover_from_uv_to_image_channelwise(
    U: torch.Tensor,
    V: torch.Tensor,
    patch: int = 8,
    img_hw: int = 128,
    clamp: bool = True,
):
    assert U.ndim == 4 and V.ndim == 4, "U, V must be (B,3,N,r) and (B,3,d,r)"
    B, C, N, r = U.shape
    assert C == 3, f"Expected 3 channels, got {C}"
    nh = nw = img_hw // patch
    assert N == nh * nw, f"Expected N={nh*nw}, got {N}"
    d = patch * patch
    assert V.shape == (B, C, d, r), f"V shape mismatch: {V.shape}"

    A_hat = torch.bmm(
        U.reshape(B * C, N, r),
        V.reshape(B * C, d, r).transpose(1, 2),
    ).reshape(B, C, nh, nw, patch, patch)

    x_hat = (
        A_hat.permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .reshape(B, C, img_hw, img_hw)
    )

    if clamp:
        x_hat = x_hat.clamp(0.0, 1.0)
    return x_hat


def recover_from_alpha_to_image(
    alpha: torch.Tensor,
    D: torch.Tensor,
    mean: torch.Tensor,
    patch: int = 8,
    img_hw: int = 128,
    clamp: bool = True,
):
    B, C, N, r = alpha.shape
    nh = nw = img_hw // patch

    D_t = D.transpose(-1, -2)
    A_hat = torch.matmul(alpha, D_t)
    A_hat = A_hat + mean[None, :, None, :]

    x_hat = (
        A_hat.reshape(B, C, nh, nw, patch, patch)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .reshape(B, C, img_hw, img_hw)
    )

    if clamp:
        x_hat = x_hat.clamp(0.0, 1.0)
    return x_hat


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob.glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    global_dict_D = None
    global_dict_mean = None
    alpha_rank_std = None
    vhat_std = None

    if args.use_global_dict:
        assert args.dict_path, "--dict-path is required when --use-global-dict is set"
        ckpt_dict = torch.load(args.dict_path, map_location="cpu", weights_only=False)
        global_dict_D = ckpt_dict["D"].float()
        _raw_mean = ckpt_dict.get("mean", None)
        global_dict_mean = _raw_mean.float() if _raw_mean is not None else None
        mean_info = (
            f"mean {tuple(global_dict_mean.shape)}"
            if global_dict_mean is not None else "mean=None (not in dict file)"
        )
        logger.info(
            f"Loaded global dict from {args.dict_path}: "
            f"D {tuple(global_dict_D.shape)}, {mean_info}"
        )

    if args.alpha_stats_path:
        stats = torch.load(args.alpha_stats_path, map_location="cpu", weights_only=False)
        alpha_rank_std = stats["std"].float()
        logger.info(
            f"Loaded per-rank std from {args.alpha_stats_path}: "
            f"shape {tuple(alpha_rank_std.shape)}, "
            f"range [{alpha_rank_std.min():.4f}, {alpha_rank_std.max():.4f}]"
        )
    else:
        logger.info(f"No --alpha-stats-path provided; fallback to scalar --norm-std={args.norm_std}")

    if args.proc_align and args.vhat_stats_path:
        vhat_ckpt = torch.load(args.vhat_stats_path, map_location="cpu", weights_only=False)
        vhat_std = vhat_ckpt["std"].float()
        logger.info(f"Loaded V_hat std from {args.vhat_stats_path}: {vhat_std:.6f}")
    elif args.proc_align:
        logger.info("No --vhat-stats-path provided; V_hat will NOT be normalized.")

    if args.proc_align:
        ref_anchor_path = os.path.join(args.shard_dir, "ref_anchor.pt")
        if os.path.exists(ref_anchor_path):
            ref_anchor = torch.load(ref_anchor_path, map_location="cpu", weights_only=False)
            global_dict_mean = ref_anchor["mean_ref"].float()
            logger.info(
                f"proc-align: loaded mean_ref from {ref_anchor_path} "
                f"(shape={tuple(global_dict_mean.shape)}) -- used for visualization."
            )
        else:
            logger.warning(
                f"proc-align: ref_anchor.pt not found at {ref_anchor_path}; "
                "visualization will use zeros/global_dict_mean fallback."
            )

    if args.proc_align:
        dataset = ShardedProcAlphaDataset(args.shard_dir, preload=False)
    elif args.use_global_dict:
        dataset = ShardedAlphaDataset(args.shard_dir, preload=False)
    else:
        dataset = ShardedUVDataset(args.shard_dir, mode="uv", preload=False, device=None)
        
    assert len(dataset) >= 100, "you should have at least 100 images in your folder. at least 10k images recommended"
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size,
        shuffle=True,
        pin_memory=True,
        pin_memory_device="cuda",
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    joint_layout = None
    if args.model == "JointDiT":
        joint_layout = infer_joint_layout(
            input_size=args.input_size,
            patch_size=args.patch_size,
            svd_rank=args.svd_rank,
        )
        logger.info(
            "Joint layout inferred: "
            f"img_hw={joint_layout['img_hw']}, "
            f"patch={joint_layout['patch']}, "
            f"rank={joint_layout['rank']}, "
            f"alpha_n={joint_layout['alpha_n']}, "
            f"patch_dim={joint_layout['patch_dim']}, "
            f"joint=({joint_layout['joint_h']}, {joint_layout['joint_w']})"
        )

        if args.proc_align:
            first_shard = dataset.shard_paths[0]
            sample = torch.load(first_shard, map_location="cpu", weights_only=False)
            alpha_shape = tuple(sample["alpha"].shape)
            vhat_shape = tuple(sample["V_hat"].shape)
            assert alpha_shape[2] == joint_layout["alpha_n"], (
                f"alpha N mismatch: shard has {alpha_shape[2]}, inferred {joint_layout['alpha_n']}"
            )
            assert vhat_shape[2] == joint_layout["patch_dim"], (
                f"V_hat patch_dim mismatch: shard has {vhat_shape[2]}, inferred {joint_layout['patch_dim']}"
            )
            assert alpha_shape[3] == joint_layout["rank"] == vhat_shape[3], (
                f"rank mismatch: alpha {alpha_shape[3]}, V_hat {vhat_shape[3]}, inferred {joint_layout['rank']}"
            )
            logger.info(
                f"Shard sanity check passed: alpha {alpha_shape}, V_hat {vhat_shape}, first shard={os.path.basename(first_shard)}"
            )

    if args.model == "DiT":
        model = DiT(
            input_size=args.input_size,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            class_dropout_prob=args.class_dropout_prob,
            use_col_mask=args.use_col_mask,
            radius=args.radius,
            learn_sigma=args.learn_sigma,
            dict_D=global_dict_D,
            dict_mean=global_dict_mean,
            use_dict_cond=args.use_dict_cond,
            img_patch_grid=tuple(args.img_patch_grid) if args.img_patch_grid else None,
            channel_wise_embed=args.channel_wise_embed,
        ).to(device)
    elif args.model == "JointDiT":
        assert joint_layout is not None
        model = JointDiT(
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            img_size=joint_layout["img_hw"],
            patch_size=joint_layout["patch"],
            rank=joint_layout["rank"],
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}. Supported: DiT, JointDiT.")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def find_latest_ckpt(path_or_dir: str) -> str:
        if os.path.isdir(path_or_dir):
            cands = sorted(glob.glob(os.path.join(path_or_dir, "*.pt")))
            if not cands:
                raise FileNotFoundError(f"No .pt in {path_or_dir}")
            return cands[-1]
        return path_or_dir

    def _strip_compile_prefix(sd: dict) -> dict:
        return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    start_epoch = 0
    train_steps = 0
    resume_ckpt = None
    if args.resume:
        ckpt_path = find_latest_ckpt(args.resume)
        print(f"[Resume] Loading checkpoint: {ckpt_path}")
        resume_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(_strip_compile_prefix(resume_ckpt["model"]), strict=True)
        opt.load_state_dict(resume_ckpt["opt"])
        start_epoch = int(resume_ckpt.get("epoch", 0))
        train_steps = int(resume_ckpt.get("train_steps", 0))

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    if resume_ckpt is not None and resume_ckpt.get("ema") is not None:
        ema.load_state_dict(_strip_compile_prefix(resume_ckpt["ema"]), strict=True)
    del resume_ckpt

    model = torch.compile(model, mode="reduce-overhead")
    model = model.to(memory_format=torch.channels_last)

    diffusion = create_diffusion(
        learn_sigma=args.learn_sigma,
        timestep_respacing="",
        predict_xstart=True,
        use_aux_loss=args.aux_loss,
        aux_weight=args.aux_weight,
        channel_wise=args.channel_wise,
        use_global_dict=args.use_global_dict,
        global_dict_D=global_dict_D,
        global_dict_mean=global_dict_mean,
    )
    fast_diffusion = create_diffusion(
        learn_sigma=args.learn_sigma,
        timestep_respacing="250",
        predict_xstart=True,
        use_aux_loss=args.aux_loss,
        aux_weight=args.aux_weight,
        channel_wise=args.channel_wise,
        use_global_dict=args.use_global_dict,
        global_dict_D=global_dict_D,
        global_dict_mean=global_dict_mean,
    )

    logger.info(f"{args.model} Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"args: {args}")

    if args.ema:
        update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    log_steps = 0
    running_loss = 0
    start_time = time()

    auto_name = (
        f"{args.model}_"
        f"lr{args.lr}_"
        f"patch{args.patch_size[0]}x{args.patch_size[1]}_"
        f"svd{args.svd_rank}_"
        f"learnsigma{args.learn_sigma}_"
        f"UV{args.uv}_"
        f"augmentation{args.augmentation}_"
        f"channelwise{args.channel_wise}_changeloss"
    )
    name = args.wandb_run_name or auto_name
    run = wandb.init(
        entity=args.wandb_entity or None,
        project=args.wandb_project,
        name=name,
        config={
            "model": args.model,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "global_batch_size": args.global_batch_size,
            "svd_rank": args.svd_rank,
            "patch_size": args.patch_size,
            "learn_sigma": args.learn_sigma,
            "image_size": args.image_size,
            "aux_loss": args.aux_loss,
            "aux_weight": args.aux_weight,
            "channel_wise": args.channel_wise,
            "norm_std": args.norm_std,
            "use_global_dict": args.use_global_dict,
            "dict_path": args.dict_path,
        },
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/loss_epoch", step_metric="epoch")
    logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        epoch_loss_sum_tensor = torch.zeros(1, device=device)
        epoch_batches = 0
        epoch_start_time = time()

        for rdata in tqdm(loader):
            if args.proc_align:
                alpha_raw = rdata[0].to(device)
                V_hat_raw = rdata[1].to(device)

                if alpha_rank_std is not None:
                    std_dev = alpha_rank_std.to(device)[None, :, None, :]
                    alpha_norm = alpha_raw / std_dev
                else:
                    alpha_norm = alpha_raw / args.norm_std

                V_hat_norm = V_hat_raw / vhat_std.to(device) if vhat_std is not None else V_hat_raw

                if args.ortho_augment:
                    B_aug = alpha_norm.shape[0]
                    R_aug = alpha_norm.shape[3]
                    Q, _ = torch.linalg.qr(torch.randn(B_aug, R_aug, R_aug, device=device))
                    N_a = alpha_norm.shape[2]
                    N_v = V_hat_norm.shape[2]
                    alpha_norm = torch.bmm(
                        alpha_norm.reshape(B_aug, -1, R_aug), Q
                    ).reshape(B_aug, 3, N_a, R_aug)
                    V_hat_norm = torch.bmm(
                        V_hat_norm.reshape(B_aug, -1, R_aug), Q
                    ).reshape(B_aug, 3, N_v, R_aug)

                data = torch.cat([alpha_norm, V_hat_norm], dim=2)

            elif args.use_global_dict:
                raw = rdata[0].to(device)
                if args.gen_dict:
                    B_batch = raw.shape[0]
                    if alpha_rank_std is not None:
                        std_dev = alpha_rank_std.to(device)[None, :, None, :]
                        alpha_norm = raw / std_dev
                    else:
                        alpha_norm = raw / args.norm_std
                    D_batch = global_dict_D.unsqueeze(0).expand(B_batch, -1, -1, -1).to(device)
                    data = torch.cat([alpha_norm, D_batch], dim=2)
                else:
                    if alpha_rank_std is not None:
                        std_dev = alpha_rank_std.to(device)[None, :, None, :]
                        data = raw / std_dev
                    else:
                        data = raw / args.norm_std
            else:
                U, V, _ = rdata
                U = U.to(device)
                V = V.to(device)
                if args.channel_wise:
                    data = torch.cat([U, V], dim=2)
                else:
                    data = torch.cat([U.unsqueeze(1), V.unsqueeze(1)], dim=2)

            if args.augmentation and (not args.proc_align):
                data = augmentation(data)
            data = data.to(memory_format=torch.channels_last)

            t = torch.randint(0, diffusion.num_timesteps, (data.shape[0],), device=device)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(model, data, t, model_kwargs={})
                loss = loss_dict["loss"].mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if args.ema:
                update_ema(ema, model)

            running_loss += loss.item()
            epoch_loss_sum_tensor += loss.detach()
            log_steps += 1
            train_steps += 1
            epoch_batches += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps

                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}"
                )

                wandb.log({
                    "train/loss": avg_loss,
                    "train/steps_per_sec": steps_per_sec,
                    "train/step": train_steps,
                }, commit=False)

                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0:
                _model_for_save = getattr(model, "_orig_mod", model)
                checkpoint = {
                    "model": _model_for_save.state_dict(),
                    "ema": ema.state_dict() if args.ema else None,
                    "opt": opt.state_dict(),
                    "args": args,
                    "epoch": epoch,
                    "train_steps": train_steps,
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

                model_was_training = model.training
                model.eval()
                try:
                    with torch.no_grad():
                        B_vis = 4
                        H, W = args.input_size
                        z = torch.randn(B_vis, args.in_channels, H, W, device=device)
                        z = z.contiguous(memory_format=torch.channels_last)

                        samples = fast_diffusion.p_sample_loop(
                            ema.forward if args.ema else model.forward,
                            z.shape,
                            z,
                            clip_denoised=False,
                            model_kwargs={},
                            progress=False,
                            device=device,
                        )

                        if args.model == "JointDiT":
                            assert joint_layout is not None
                            alpha_n = joint_layout["alpha_n"]
                            patch_dim = joint_layout["patch_dim"]
                            img_hw_vis = joint_layout["img_hw"]
                            patch_vis = joint_layout["patch"]
                        else:
                            alpha_n = None
                            patch_dim = None
                            img_hw_vis = args.image_size
                            patch_vis = args.patch_size[0]

                        if args.proc_align:
                            alpha_samp = samples[:, :, :alpha_n, :]
                            V_samp = samples[:, :, alpha_n:, :]

                            if alpha_rank_std is not None:
                                std_dev = alpha_rank_std.to(device)[None, :, None, :]
                                alpha_samp = alpha_samp * std_dev
                            else:
                                alpha_samp = alpha_samp * args.norm_std

                            if vhat_std is not None:
                                V_samp = V_samp * vhat_std.to(device)

                            D_vis = V_samp  # (B, 3, patch_dim, rank), per-sample dictionary
                            mean_vis = (
                                global_dict_mean.to(device)
                                if global_dict_mean is not None
                                else torch.zeros(3, patch_dim, device=device)
                            )
                            samples = recover_from_alpha_to_image(
                                alpha_samp, D_vis, mean_vis, patch=patch_vis, img_hw=img_hw_vis
                            )

                        elif args.gen_dict:
                            alpha_samp = samples[:, :, :alpha_n, :]
                            D_samp_raw = samples[:, :, alpha_n:, :]

                            if alpha_rank_std is not None:
                                std_dev = alpha_rank_std.to(device)[None, :, None, :]
                                alpha_samp = alpha_samp * std_dev
                            else:
                                alpha_samp = alpha_samp * args.norm_std

                            D_vis = D_samp_raw  # (B, 3, patch_dim, rank), per-sample dictionary
                            mean_vis = (
                                global_dict_mean.to(device)
                                if global_dict_mean is not None
                                else torch.zeros(3, patch_dim, device=device)
                            )
                            samples = recover_from_alpha_to_image(
                                alpha_samp, D_vis, mean_vis, patch=patch_vis, img_hw=img_hw_vis
                            )

                        elif args.use_global_dict and alpha_rank_std is not None:
                            std_dev = alpha_rank_std.to(device)[None, :, None, :]
                            samples = samples * std_dev
                            D_vis = global_dict_D.to(device)
                            mean_vis = (
                                global_dict_mean.to(device)
                                if global_dict_mean is not None
                                else torch.zeros(3, global_dict_D.shape[1], device=device)
                            )
                            samples = recover_from_alpha_to_image(
                                samples, D_vis, mean_vis, patch=patch_vis, img_hw=img_hw_vis
                            )

                        elif args.use_global_dict:
                            samples = samples * args.norm_std
                            D_vis = global_dict_D.to(device)
                            mean_vis = (
                                global_dict_mean.to(device)
                                if global_dict_mean is not None
                                else torch.zeros(3, global_dict_D.shape[1], device=device)
                            )
                            samples = recover_from_alpha_to_image(
                                samples, D_vis, mean_vis, patch=patch_vis, img_hw=img_hw_vis
                            )

                        elif args.channel_wise:
                            U_vis = samples[:, :, :alpha_n, :]
                            V_vis = samples[:, :, alpha_n:, :]
                            samples = recover_from_uv_to_image_channelwise(
                                U_vis, V_vis, patch=patch_vis, img_hw=img_hw_vis
                            )

                        else:
                            samples = samples.squeeze(1)
                            U_vis = samples[:, :256, :]
                            V_vis = samples[:, 256:, :]
                            samples = recover_from_uv_to_image(U_vis, V_vis, patch=patch_vis, img_hw=img_hw_vis)

                        q95 = torch.max(samples)
                        q05 = torch.min(samples)
                        grid = vutils.make_grid(
                            samples, nrow=4, normalize=True, value_range=(q05.item(), q95.item())
                        )

                        wandb.log({
                            "samples/checkpoint_preview": wandb.Image(
                                grid.permute(1, 2, 0).detach().cpu().numpy(),
                                caption=f"step {train_steps}"
                            )
                        })
                except Exception as e:
                    logger.warning(f"Sampling or W&B logging failed at step {train_steps}: {e}")
                finally:
                    if model_was_training:
                        model.train()

        epoch_avg_loss = (epoch_loss_sum_tensor / max(1, epoch_batches)).item()
        epoch_duration = time() - epoch_start_time

        wandb.log({
            "epoch": epoch,
            "train/loss_epoch": epoch_avg_loss,
            "train/time_epoch": epoch_duration
        }, commit=True)

        wandb.run.summary["last_train_loss_epoch"] = epoch_avg_loss

    model.eval()

    _model_for_save = getattr(model, "_orig_mod", model)
    checkpoint = {
        "model": _model_for_save.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "args": args,
        "epoch": epoch,
        "train_steps": train_steps,
    }
    checkpoint_path = f"{checkpoint_dir}/final.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved final checkpoint to {checkpoint_path}")

    logger.info("Done!")
    cleanup()


def _load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(cfg).__name__}")
    return cfg


if __name__ == "__main__":
    # ---- Pre-parse to discover --config so YAML can populate parser defaults ----
    pre = argparse.ArgumentParser(add_help=False)
    _default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.yaml")
    pre.add_argument("--config", type=str,
                     default=_default_cfg if os.path.exists(_default_cfg) else None,
                     help="Path to YAML config file (default: train.yaml next to train.py)")
    pre_args, _ = pre.parse_known_args()

    yaml_cfg = _load_yaml_config(pre_args.config)
    if pre_args.config:
        print(f"[Config] Loaded {pre_args.config} ({len(yaml_cfg)} keys)")

    parser = argparse.ArgumentParser(parents=[pre])

    parser.add_argument("--hidden-size", type=int, default=1152)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)

    parser.add_argument("--svd_rank", type=int, default=32,
                        help="If >0, train in SVD mode using rank-r packing.")
    parser.add_argument("--use_col_mask", action="store_true",
                        help="Enable column neighbor attention mask")
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--learn-sigma", action="store_true", default=False, help="Enable learn_sigma")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, choices=["DiT", "JointDiT"], default="JointDiT")

    parser.add_argument("--gen-dict", action="store_true", default=False,
                        help="Choice B: jointly generate alpha and D/V_hat.")
    parser.add_argument("--proc-align", action="store_true", default=False,
                        help="Procrustes-aligned per-image SVD mode.")
    parser.add_argument("--vhat-stats-path", type=str, default="${DATA_ROOT}/tucker_factors/celeba/Exp_p32r32_acceleration/vhat_stats_procrustes_refimg_p32_r32.pt",
                        help="Path to V_hat normalization stats .pt file {'std': scalar}.")
    parser.add_argument("--wandb_project", type=str, default="myproject")
    parser.add_argument("--wandb-run-name", type=str, default="",
                        help="W&B run name; empty -> auto-generate from hyperparameters")
    parser.add_argument("--wandb-entity", type=str, default="",
                        help="W&B entity; empty -> use default from ~/.netrc")
    parser.add_argument("--ema", action="store_true", default=False, help="Enable EMA")
    parser.add_argument("--uv", action="store_true", default=False, help="Enable UV")
    parser.add_argument("--augmentation", action="store_true", default=False, help="Enable data augmentation")
    parser.add_argument("--ortho-augment", action="store_true", default=False,
                        help="Per-image random orthogonal augmentation on (alpha, V_hat).")
    parser.add_argument("--shard-dir", type=str,
                        default="${DATA_ROOT}/tucker_factors/celeba/Exp_p32r32_acceleration/celebahq1024_patchsvd_procrustes_refimg_p32_r32",
                        help="Directory containing pre-computed shard .pt files")

    parser.add_argument("--aux-loss", action="store_true", default=False,
                        help="Add auxiliary loss in reconstructed image space")
    parser.add_argument("--aux-weight", type=float, default=0.001,
                        help="Weight of auxiliary loss term")
    parser.add_argument("--channel-wise", action="store_true", default=False,
                        help="Use channel-wise patch-SVD data")
    parser.add_argument("--channel-wise-embed", action="store_true", default=False,
                        help="Use channel-wise patch embed in DiT")
    parser.add_argument("--use-global-dict", action="store_true", default=False,
                        help="Use global PCA dict")
    parser.add_argument("--use-dict-cond", action="store_true", default=True,
                        help="Use cross-attention conditioning on dictionary tokens")
    parser.add_argument("--dict-path", type=str, default="",
                        help="Path to global dict checkpoint containing D and optional mean")
    parser.add_argument("--alpha-stats-path", type=str, default="${DATA_ROOT}/tucker_factors/celeba/Exp_p32r32_acceleration/alpha_stats_procrustes_refimg_p32_r32.pt",
                        help="Path to alpha normalization stats")
    parser.add_argument("--norm-std", type=float, default=0.5,
                        help="Fallback scalar normalization std")
    parser.add_argument("--results_dir", type=str, default="${DATA_ROOT}/ablation_results")
    parser.add_argument("--resume", type=str, default="",
                        help="Checkpoint path or checkpoint dir to resume from")

    parser.add_argument("--input-size", type=int, nargs=2, default=[2048, 32],
                        help="Model input size. For JointDiT 1024/p32/r32 use 2048 32; p32/r16 use 2048 16; p16/r16 use 4352 16.")
    parser.add_argument("--patch-size", type=int, nargs=2, default=[32, 32],
                        help="Patch size used by DiT / layout inference")
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512, 1024], default=1024)
    parser.add_argument("--img-patch-grid", type=int, nargs=2, default=None)

    parser.set_defaults(proc_align=True, use_global_dict=False, gen_dict=False, channel_wise=False, augmentation=False)

    # ---- YAML overrides hardcoded defaults; CLI overrides YAML ----
    if yaml_cfg:
        known_dests = {a.dest for a in parser._actions}
        # Documented in YAML for reproducibility but not consumed by argparse
        # (hard-coded inside main() or only describing derived values).
        documented_only = {
            "patch", "prefetch_factor", "compile", "use_bf16",
            "noise_schedule", "diffusion_steps", "sample_steps", "predict_xstart",
        }
        filtered, unknown = {}, []
        for k, v in yaml_cfg.items():
            if k in known_dests:
                filtered[k] = v
            elif k in documented_only:
                continue
            else:
                unknown.append(k)
        if unknown:
            print(f"[Config] Ignoring unknown YAML keys: {unknown}")
        if filtered:
            parser.set_defaults(**filtered)

    args = parser.parse_args()

    if args.model == "JointDiT":
        layout = infer_joint_layout(args.input_size, args.patch_size, args.svd_rank)
        if args.image_size != layout["img_hw"]:
            print(
                f"[WARN] args.image_size={args.image_size} != inferred JointDiT image size {layout['img_hw']}. "
                f"Preview reconstruction will use inferred value."
            )

    main(args)