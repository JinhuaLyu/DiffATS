#!/usr/bin/env python3
import argparse
import glob
import json
import os
import random
import time
from datetime import datetime
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import inception_v3
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ───────── InceptionV3 pool3 features (2048-D) ─────────

class InceptionPool3(torch.nn.Module):
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


# ───────── DINOv2 ViT-L/14 CLS features (1024-D) ─────────

class DinoV2Embed(torch.nn.Module):
    def __init__(self, model_name: str = "dinov2_vitl14", img_size: int = 224):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", model_name, trust_repo=True
        )
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


# ───────── I/O ─────────

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


def extract_feats(paths, model, device, batch_size, desc="extracting") -> np.ndarray:
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc=f"  {desc}"):
        batch = torch.stack(
            [load_image(p) for p in paths[i: i + batch_size]]
        ).to(device, non_blocking=True)
        feats.append(model(batch).float().cpu())
    return torch.cat(feats).numpy()


# ───────── FID (rank-trick, GPU, fp64) ─────────

@torch.no_grad()
def fid_from_feats_gpu(real: np.ndarray, fake: np.ndarray, device,
                       dtype: torch.dtype = torch.float64) -> float:
    R  = torch.from_numpy(real).to(device=device, dtype=dtype)
    Fm = torch.from_numpy(fake).to(device=device, dtype=dtype)
    n1, n2 = R.shape[0], Fm.shape[0]

    mu1, mu2 = R.mean(0), Fm.mean(0)
    Rc, Fc   = R - mu1,   Fm - mu2
    del R, Fm

    diff    = mu1 - mu2
    tr_sig1 = (Rc * Rc).sum() / (n1 - 1)
    tr_sig2 = (Fc * Fc).sum() / (n2 - 1)

    A       = Rc @ Fc.T
    del Rc, Fc
    sigvals  = torch.linalg.svdvals(A)
    tr_sqrt  = sigvals.sum() / float((n1 - 1) * (n2 - 1)) ** 0.5

    return float((diff @ diff + tr_sig1 + tr_sig2 - 2.0 * tr_sqrt).item())


# ───────── Precision & Recall (k-NN manifold, InceptionV3) ─────────

def manifold_radii(feats: np.ndarray, k: int, device) -> torch.Tensor:
    X = torch.from_numpy(feats).to(device)
    N = X.shape[0]
    radii = torch.empty(N, device=device)
    chunk = 1024
    for i in range(0, N, chunk):
        Xi   = X[i: i + chunk]
        d    = torch.cdist(Xi, X)
        rows = torch.arange(Xi.shape[0], device=device) + i
        d[torch.arange(Xi.shape[0], device=device), rows] = float("inf")
        topk = torch.topk(d, k=k, dim=1, largest=False).values
        radii[i: i + Xi.shape[0]] = topk[:, -1]
    return radii


def precision_recall(real_feats: np.ndarray, fake_feats: np.ndarray,
                     k: int, device) -> Tuple[float, float]:
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


# ───────── main ─────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-dir",   required=True,
        help="Directory with generated PNG/JPG images")
    p.add_argument("--orig-dir",
        default="${DATA_ROOT}/original_data/celeba",
        help="Directory with real CelebA-HQ images")
    p.add_argument("--n-samples",  type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--knn-k",      type=int, default=3,
        help="k for Precision/Recall k-NN manifold")
    p.add_argument("--dino-model", default="dinov2_vitl14",
        choices=["dinov2_vits14", "dinov2_vitb14",
                 "dinov2_vitl14", "dinov2_vitg14"])
    p.add_argument("--dino-img-size", type=int, default=224)
    p.add_argument("--label",      default="gen",
        help="Label for this run in the output JSON")
    p.add_argument("--out-json",   required=True)
    p.add_argument("--skip-dino",  action="store_true",
        help="Skip DINOv2 FID (faster, InceptionV3 only)")
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    print(f"[INFO] device       = {device}")
    print(f"[INFO] gen_dir      = {args.gen_dir}")
    print(f"[INFO] orig_dir     = {args.orig_dir}")
    print(f"[INFO] n_samples    = {args.n_samples}")
    print(f"[INFO] knn_k        = {args.knn_k}")
    print(f"[INFO] skip_dino    = {args.skip_dino}")

    # collect paths
    real_paths = collect_paths(args.orig_dir, args.n_samples, args.seed)
    gen_paths  = collect_paths(args.gen_dir,  args.n_samples, args.seed)
    n_avail    = len([f for f in os.listdir(args.gen_dir)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"[INFO] real images  = {len(real_paths)}")
    print(f"[INFO] gen images   = {len(gen_paths)} / {n_avail} available")

    results = {}

    # ── Step 1: InceptionV3 FID + Precision/Recall ──────────────────
    print("\n[Step 1] InceptionV3 features (2048-D) ...")
    incep_model = InceptionPool3().to(device).eval()

    print("  [REAL] extracting InceptionV3 features ...")
    real_incep = extract_feats(real_paths, incep_model, device,
                               args.batch_size, desc="real/inception")
    print("  [GEN]  extracting InceptionV3 features ...")
    gen_incep  = extract_feats(gen_paths,  incep_model, device,
                               args.batch_size, desc="gen/inception")

    print("  [FID_InceptionV3] computing ...")
    fid_incep = fid_from_feats_gpu(real_incep, gen_incep, device)
    print(f"  FID_InceptionV3 = {fid_incep:.4f}")

    print("  [Precision & Recall] computing ...")
    prec, rec = precision_recall(real_incep, gen_incep, args.knn_k, device)
    print(f"  Precision = {prec:.4f}   Recall = {rec:.4f}")

    results["FID_InceptionV3"] = round(fid_incep, 4)
    results["Precision"]       = round(prec,      4)
    results["Recall"]          = round(rec,        4)

    del incep_model, real_incep, gen_incep
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Step 2: DINOv2 FID ──────────────────────────────────────────
    if not args.skip_dino:
        print(f"\n[Step 2] DINOv2 features ({args.dino_model}, {args.dino_img_size}px) ...")
        dino_model = DinoV2Embed(
            model_name=args.dino_model,
            img_size=args.dino_img_size
        ).to(device).eval()

        print("  [REAL] extracting DINOv2 features ...")
        real_dino = extract_feats(real_paths, dino_model, device,
                                  args.batch_size, desc="real/dino")
        print("  [GEN]  extracting DINOv2 features ...")
        gen_dino  = extract_feats(gen_paths,  dino_model, device,
                                  args.batch_size, desc="gen/dino")

        print("  [FID_DINOv2] computing ...")
        fid_dino = fid_from_feats_gpu(real_dino, gen_dino, device)
        print(f"  FID_DINOv2 = {fid_dino:.4f}")

        results["FID_DINOv2"] = round(fid_dino, 4)

        del dino_model, real_dino, gen_dino
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("\n[Step 2] Skipping DINOv2 (--skip-dino flag set)")

    # ── Save JSON ───────────────────────────────────────────────────
    total_time = time.time() - t0
    out = {
        "config": {
            "gen_dir":          args.gen_dir,
            "orig_dir":         args.orig_dir,
            "n_samples":        args.n_samples,
            "n_real_used":      len(real_paths),
            "n_gen_used":       len(gen_paths),
            "n_gen_available":  n_avail,
            "batch_size":       args.batch_size,
            "seed":             args.seed,
            "knn_k":            args.knn_k,
            "dino_model":       args.dino_model,
            "dino_img_size":    args.dino_img_size,
            "skip_dino":        args.skip_dino,
            "fid_backend":      "gpu_rank_trick_fp64",
            "total_time_min":   round(total_time / 60, 2),
            "timestamp":        datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": {
            args.label: results
        },
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[DONE] saved -> {args.out_json}  (total {total_time/60:.1f} min)")
    print("\n=== Summary ===")
    print(f"label            : {args.label}")
    for k, v in results.items():
        print(f"{k:<20} : {v:.4f}")


if __name__ == "__main__":
    main()
