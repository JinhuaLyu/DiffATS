#!/usr/bin/env python3
"""
Compute FID (InceptionV3) + FID (DINOv2) + Precision/Recall for one or
many generated-image directories against a reference dataset.

Two modes (mutually exclusive):
  - Single  : --gen-dir <path> --label <name>
  - Batch   : --gen-spec name1=path1 name2=path2 ...

Metrics per directory:
  FID_Inception, FID_DINOv2, Precision, Recall

FID uses the rank-trick on GPU in fp64
  trace(sqrt(Sig1 Sig2)) = ||Rc Fc^T||_* / sqrt((n1-1)(n2-1))
which is much faster than scipy.linalg.sqrtm on the full covariance.

Precision/Recall is the improved version (Kynkaanniemi 2019) on Inception
pool3 features with k-NN manifold radii (default k=3).

DINOv2 FID uses CLS-token embeddings of the chosen ViT (default ViT-L/14),
following Stein et al. 2023.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import inception_v3
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# --------- Feature extractors ---------
class InceptionPool3(torch.nn.Module):
    """InceptionV3 pool3 features (2048-D) used for FID and P&R."""

    def __init__(self):
        super().__init__()
        m = inception_v3(weights="IMAGENET1K_V1", transform_input=False, aux_logits=True)
        self.backbone = torch.nn.Sequential(
            m.Conv2d_1a_3x3, m.Conv2d_2a_3x3, m.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(3, 2),
            m.Conv2d_3b_1x1, m.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(3, 2),
            m.Mixed_5b, m.Mixed_5c, m.Mixed_5d,
            m.Mixed_6a, m.Mixed_6b, m.Mixed_6c, m.Mixed_6d, m.Mixed_6e,
            m.Mixed_7a, m.Mixed_7b, m.Mixed_7c, m.avgpool,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = x * 2.0 - 1.0
        return self.backbone(x).flatten(1)


class DinoV2Embed(torch.nn.Module):
    """DINOv2 ViT CLS-token embedding (default ViT-L/14, 1024-D)."""

    def __init__(self, model_name: str = "dinov2_vitl14", img_size: int = 224):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.img_size = img_size
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD ).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bicubic", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        return self.backbone(x)


# --------- I/O ---------
def collect_paths(directory: str, n: int, seed: int) -> List[str]:
    paths = sorted(
        glob.glob(os.path.join(directory, "*.png"))
        + glob.glob(os.path.join(directory, "*.jpg"))
        + glob.glob(os.path.join(directory, "*.jpeg"))
    )
    if not paths:
        raise FileNotFoundError(f"No images in {directory}")
    rng = random.Random(seed)
    if len(paths) > n:
        paths = rng.sample(paths, n)
    return paths


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def extract_feats(paths, model, device, batch_size) -> np.ndarray:
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="  extracting"):
        batch = torch.stack([load_image(p) for p in paths[i: i + batch_size]]).to(device, non_blocking=True)
        feats.append(model(batch).float().cpu())
    return torch.cat(feats).numpy()


# --------- FID (rank-trick, GPU, fp64) ---------
@torch.no_grad()
def fid_from_feats_gpu(real: np.ndarray, fake: np.ndarray, device,
                       dtype: torch.dtype = torch.float64) -> float:
    R  = torch.from_numpy(real).to(device=device, dtype=dtype)
    Fm = torch.from_numpy(fake).to(device=device, dtype=dtype)
    n1, n2 = R.shape[0], Fm.shape[0]

    mu1, mu2 = R.mean(0), Fm.mean(0)
    Rc, Fc = R - mu1, Fm - mu2
    del R, Fm

    diff = mu1 - mu2
    tr_sig1 = (Rc * Rc).sum() / (n1 - 1)
    tr_sig2 = (Fc * Fc).sum() / (n2 - 1)

    A = Rc @ Fc.T
    del Rc, Fc
    sigvals = torch.linalg.svdvals(A)
    tr_sqrt = sigvals.sum() / float((n1 - 1) * (n2 - 1)) ** 0.5
    return float((diff @ diff + tr_sig1 + tr_sig2 - 2.0 * tr_sqrt).item())


# --------- Precision / Recall (Kynkaanniemi 2019) ---------
def manifold_radii(feats: np.ndarray, k: int, device) -> torch.Tensor:
    X = torch.from_numpy(feats).to(device)
    N = X.shape[0]
    radii = torch.empty(N, device=device)
    chunk = 1024
    for i in range(0, N, chunk):
        Xi = X[i: i + chunk]
        d = torch.cdist(Xi, X)
        rows = torch.arange(Xi.shape[0], device=device) + i
        d[torch.arange(Xi.shape[0], device=device), rows] = float("inf")
        topk = torch.topk(d, k=k, dim=1, largest=False).values
        radii[i: i + Xi.shape[0]] = topk[:, -1]
    return radii


def precision_recall(real_feats: np.ndarray, fake_feats: np.ndarray, k: int) -> Tuple[float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_radii = manifold_radii(real_feats, k=k, device=device)
    fake_radii = manifold_radii(fake_feats, k=k, device=device)
    R  = torch.from_numpy(real_feats).to(device)
    F_ = torch.from_numpy(fake_feats).to(device)

    chunk = 1024
    Nr, Nf = R.shape[0], F_.shape[0]

    in_real = torch.zeros(Nf, dtype=torch.bool, device=device)
    for i in range(0, Nf, chunk):
        d = torch.cdist(F_[i: i + chunk], R)
        in_real[i: i + chunk] = (d <= real_radii.unsqueeze(0)).any(dim=1)
    precision = in_real.float().mean().item()

    in_fake = torch.zeros(Nr, dtype=torch.bool, device=device)
    for i in range(0, Nr, chunk):
        d = torch.cdist(R[i: i + chunk], F_)
        in_fake[i: i + chunk] = (d <= fake_radii.unsqueeze(0)).any(dim=1)
    recall = in_fake.float().mean().item()

    return precision, recall


# --------- Per-directory pipeline ---------
def evaluate_one(name: str, gen_dir: str,
                 real_paths: List[str],
                 real_p3: np.ndarray, real_dino: np.ndarray | None,
                 inception, dino, device, args) -> Dict:
    if not os.path.isdir(gen_dir):
        return {"error": "dir missing", "gen_dir": gen_dir}
    n_avail = len([p for p in os.listdir(gen_dir)
                   if p.lower().endswith((".png", ".jpg", ".jpeg"))])
    if n_avail == 0:
        return {"error": "no images", "gen_dir": gen_dir}

    gen_paths = collect_paths(gen_dir, args.n_samples, args.seed)
    print(f"\n[{name}] {gen_dir}  ({len(gen_paths)} of {n_avail})")

    print("  extracting Inception ...")
    gen_p3 = extract_feats(gen_paths, inception, device, args.batch_size)

    t = time.time()
    fid_inc = fid_from_feats_gpu(real_p3, gen_p3, device)
    print(f"  FID_Inception = {fid_inc:.4f}    ({time.time() - t:.1f}s)")

    t = time.time()
    prec, rec = precision_recall(real_p3, gen_p3, k=args.knn_k)
    print(f"  Precision     = {prec:.4f}")
    print(f"  Recall        = {rec:.4f}    ({time.time() - t:.1f}s for P&R)")
    del gen_p3
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out: Dict = {
        "FID_Inception": round(fid_inc, 4),
        "Precision":     round(prec,    4),
        "Recall":        round(rec,     4),
        "gen_dir":               gen_dir,
        "n_generated_used":      len(gen_paths),
        "n_generated_available": n_avail,
    }

    if dino is not None and real_dino is not None:
        print("  extracting DINOv2 ...")
        gen_d = extract_feats(gen_paths, dino, device, args.batch_size)
        t = time.time()
        fid_dino = fid_from_feats_gpu(real_dino, gen_d, device)
        print(f"  FID_DINOv2    = {fid_dino:.4f}    ({time.time() - t:.1f}s)")
        out["FID_DINOv2"] = round(fid_dino, 4)
        del gen_d
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


# --------- CLI ---------
def parse_gen_spec(items: List[str]) -> List[Tuple[str, str]]:
    out = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--gen-spec entry must be name=path, got: {item}")
        name, path = item.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig-dir", required=True,
                   help="directory of real images for FID reference")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--gen-dir", help="single generated-image directory")
    grp.add_argument("--gen-spec", nargs="+",
                     help="batch spec list: name1=path1 name2=path2 ...")
    p.add_argument("--label", default="gen",
                   help="label for the result entry in --gen-dir mode")
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--knn-k", type=int, default=3)
    p.add_argument("--skip-dinov2", action="store_true",
                   help="skip the DINOv2 FID computation")
    p.add_argument("--dinov2-model", default="dinov2_vitl14",
                   choices=["dinov2_vits14", "dinov2_vitb14",
                            "dinov2_vitl14", "dinov2_vitg14"])
    p.add_argument("--dinov2-img-size", type=int, default=224)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}, n_samples = {args.n_samples}, k = {args.knn_k}")
    print(f"[INFO] orig_dir = {args.orig_dir}")

    # Resolve mode -> list[(name, path)]
    if args.gen_dir is not None:
        gen_specs = [(args.label, args.gen_dir)]
    else:
        gen_specs = parse_gen_spec(args.gen_spec)
    for n, pth in gen_specs:
        print(f"[INFO]   gen[{n}] = {pth}")

    t0 = time.time()
    real_paths = collect_paths(args.orig_dir, args.n_samples, args.seed)
    print(f"\n[REAL] {len(real_paths)} images from {args.orig_dir}")

    print("\n[INFO] loading InceptionV3 ...")
    inception = InceptionPool3().to(device).eval()
    print("[REAL] extracting Inception pool3 ...")
    real_p3 = extract_feats(real_paths, inception, device, args.batch_size)

    real_dino = None
    dino = None
    if not args.skip_dinov2:
        print(f"\n[INFO] loading {args.dinov2_model} ...")
        dino = DinoV2Embed(model_name=args.dinov2_model,
                           img_size=args.dinov2_img_size).to(device).eval()
        print("[REAL] extracting DINOv2 ...")
        real_dino = extract_feats(real_paths, dino, device, args.batch_size)

    out = {
        "config": {
            "orig_dir":   args.orig_dir,
            "n_samples":  args.n_samples,
            "n_real_used": len(real_paths),
            "batch_size": args.batch_size,
            "seed":       args.seed,
            "knn_k":      args.knn_k,
            "fid_backend": "gpu_rank_trick_fp64",
            "inception_extractor": "torchvision_inception_v3_IMAGENET1K_V1_pool3_2048",
            "dinov2_extractor": (None if args.skip_dinov2
                                 else f"{args.dinov2_model}_cls_img{args.dinov2_img_size}"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": {},
    }

    for name, gen_dir in gen_specs:
        out["metrics"][name] = evaluate_one(
            name, gen_dir, real_paths, real_p3, real_dino,
            inception, dino, device, args,
        )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[DONE] saved -> {args.out_json}  (total {(time.time() - t0)/60:.1f} min)")

    print("\n=== Summary ===")
    hdr = f"{'name':<28} {'FID_Inc':>10} {'FID_Dino':>10} {'P':>8} {'R':>8}"
    print(hdr)
    for k, v in out["metrics"].items():
        if "error" in v:
            print(f"{k:<28} ERROR: {v['error']}")
            continue
        fid_inc  = v["FID_Inception"]
        fid_dino = v.get("FID_DINOv2", float("nan"))
        prec     = v["Precision"]
        rec      = v["Recall"]
        print(f"{k:<28} {fid_inc:>10.4f} {fid_dino:>10.4f} {prec:>8.4f} {rec:>8.4f}")


if __name__ == "__main__":
    main()
