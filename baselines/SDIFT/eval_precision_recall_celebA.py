import argparse
import glob
import json
import os
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

K_VALS = [3, 5, 10, 20]
BETA_PREC = 1 / 8.0   # F_{1/8} ≈ precision
BETA_REC  = 8.0        # F_8    ≈ recall


# ───────── DINOv2 backbone ─────────
class DinoV2Embed(torch.nn.Module):
    def __init__(self, model_name="dinov2_vitl14", img_size=224):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name,
                                       trust_repo=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.img_size = img_size
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD ).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    @torch.no_grad()
    def forward(self, x):
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bicubic", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        return self.backbone(x)


# ───────── I/O ─────────
def collect_paths(directory, n, seed):
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


def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def extract_feats(paths, model, device, batch_size):
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="  extracting"):
        batch = torch.stack([load_image(p) for p in paths[i: i + batch_size]]).to(
            device, dtype=torch.float32, non_blocking=True
        )
        feats.append(model(batch).float().cpu())
    return torch.cat(feats).numpy()


# ───────── IPR metric ─────────
def _knn_radii(feats, k):
    """k-th nearest-neighbour distance for each point (excluding self)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean",
                          algorithm="auto", n_jobs=-1)
    nn.fit(feats)
    dists, _ = nn.kneighbors(feats)
    return dists[:, k]   # index k = k-th NN (0 is self)


def _in_manifold(query_feats, ref_feats, ref_radii, k):
    """Fraction of query points that lie inside the k-NN manifold of ref."""
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean",
                          algorithm="auto", n_jobs=-1)
    nn.fit(ref_feats)
    dists, indices = nn.kneighbors(query_feats)
    # point q is in manifold if ∃ ref r : dist(q,r) ≤ radius_k(r)
    in_mfld = np.any(dists <= ref_radii[indices], axis=1)
    return in_mfld.mean()


def ipr_f_scores(real_feats, fake_feats):
    """
    Returns (max_f_prec, max_f_rec) over K_VALS, where
      max_f_prec = max_k  F_{1/8}(precision_k, recall_k)
      max_f_rec  = max_k  F_8   (precision_k, recall_k)
    """
    best_f_prec = 0.0
    best_f_rec  = 0.0

    for k in K_VALS:
        radii_real = _knn_radii(real_feats, k)
        radii_fake = _knn_radii(fake_feats, k)

        precision = _in_manifold(fake_feats, real_feats, radii_real, k)
        recall    = _in_manifold(real_feats, fake_feats, radii_fake, k)

        denom_prec = BETA_PREC**2 * precision + recall
        f_prec = ((1 + BETA_PREC**2) * precision * recall / denom_prec
                  if denom_prec > 0 else 0.0)

        denom_rec  = BETA_REC**2 * precision + recall
        f_rec  = ((1 + BETA_REC**2) * precision * recall / denom_rec
                  if denom_rec > 0 else 0.0)

        best_f_prec = max(best_f_prec, f_prec)
        best_f_rec  = max(best_f_rec,  f_rec)

    return best_f_prec, best_f_rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig-dir",   required=True)
    p.add_argument("--gen-dir",    required=True)
    p.add_argument("--recon-dir",  required=True)
    p.add_argument("--n-samples",  type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--model",      default="dinov2_vitl14")
    p.add_argument("--img-size",   type=int, default=224)
    p.add_argument("--out-json",   required=True)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device},  model = {args.model},  k_vals = {K_VALS}")

    print("[INFO] loading DINOv2 ...")
    model = DinoV2Embed(model_name=args.model, img_size=args.img_size).to(device).eval()
    t0 = time.time()

    # ── real features ──
    print(f"\n[REAL] {args.orig_dir}")
    real_paths = collect_paths(args.orig_dir, args.n_samples, args.seed)
    print(f"  using {len(real_paths)} images")
    real_feats = extract_feats(real_paths, model, device, args.batch_size)
    print(f"  real features: {real_feats.shape}")

    results = {}

    for tag, gen_dir in [("gen vs orig", args.gen_dir),
                         ("recon vs orig", args.recon_dir)]:
        print(f"\n[{tag.upper()}] {gen_dir}")
        paths = collect_paths(gen_dir, args.n_samples, args.seed)
        print(f"  using {len(paths)} images")
        fake_feats = extract_feats(paths, model, device, args.batch_size)

        t_ipr = time.time()
        print(f"  computing IPR (k={K_VALS}) ...")
        f_prec, f_rec = ipr_f_scores(real_feats, fake_feats)
        elapsed = time.time() - t_ipr
        print(f"  max F_{{1/8.0}}  (precision) = {f_prec:.4f}   ({elapsed:.1f}s)")
        print(f"  max F_8.0      (recall)    = {f_rec:.4f}")
        results[tag] = {"f_prec_1_8": round(f_prec, 4),
                        "f_rec_8":    round(f_rec, 4),
                        "gen_dir":    gen_dir,
                        "n_used":     len(paths)}
        del fake_feats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── save ──
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"k_vals": K_VALS, "beta_prec": BETA_PREC,
                   "beta_rec": BETA_REC, "results": results}, f, indent=2)
    print(f"\n[DONE] saved -> {args.out_json}  ({(time.time()-t0)/60:.1f} min)")

    # ── pretty print ──
    print("\n" + "=" * 50)
    print("recon vs orig:")
    print(f"  max F_{{1/8.0}}  (precision) = {results['recon vs orig']['f_prec_1_8']:.4f}")
    print()
    print("gen vs orig:")
    print(f"  max F_{{1/8.0}}  (precision) = {results['gen vs orig']['f_prec_1_8']:.4f}")
    print()
    print("recon vs orig:")
    print(f"  max F_8.0      (recall)    = {results['recon vs orig']['f_rec_8']:.4f}")
    print()
    print("gen vs orig:")
    print(f"  max F_8.0      (recall)    = {results['gen vs orig']['f_rec_8']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
