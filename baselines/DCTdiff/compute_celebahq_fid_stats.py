import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from tools.fid_score import save_statistics_of_path

# Same layout as configs/celebahq1024_uvit_mid_16by16.py: .../celeba_hq_images/all/*.jpg
IMG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "CelebA-HQ", "celeba_hq_images", "all")
)

OUT_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "fid_stats", "fid_stats_celebahq1024.npz"
)


def main():
    parser = argparse.ArgumentParser(description="Compute FID stats for CelebA-HQ")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    if os.path.exists(OUT_PATH):
        print(f"[SKIP] {OUT_PATH} already exists. Delete it to recompute.")
        return

    print(f"Computing Inception-v3 statistics for images in:\n  {IMG_DIR}")
    print(f"Output: {OUT_PATH}")
    print(f"Batch size: {args.batch_size}, workers: {args.num_workers}")

    save_statistics_of_path(
        IMG_DIR,
        OUT_PATH,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Done. FID reference stats saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
