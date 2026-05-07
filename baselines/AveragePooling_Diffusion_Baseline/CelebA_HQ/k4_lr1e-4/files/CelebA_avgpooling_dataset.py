import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse


class CelebAHQDataset(Dataset):

    def __init__(
        self,
        data_dir,
        pool_k=4,
        orig_size=1024,
        img_ext='.jpg',
    ):
        self.pool_k    = pool_k
        self.orig_size = orig_size
        self.lr_size   = orig_size // pool_k   # 256

        # find all images
        img_files = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith(img_ext) or f.lower().endswith('.png')
        ])
        assert len(img_files) > 0, f"No images found in {data_dir}"

        print(f"Data dir       : {data_dir}")
        print(f"Num images     : {len(img_files)}")
        print(f"Original size  : {orig_size}x{orig_size} (no resize)")
        print(f"Pool factor k  : {pool_k}  ->  {self.lr_size}x{self.lr_size}")
        print(f"Compression    : {pool_k**2}x")
        print(f"Preloading all images into RAM...")

        # load and avg pool
        to_tensor = transforms.ToTensor()   # [3, H, W] float32 in [0, 1]

        images = []
        for fname in tqdm(img_files, desc='Loading'):
            img = Image.open(os.path.join(data_dir, fname)).convert('RGB')
            t   = to_tensor(img).unsqueeze(0)                  # [1, 3, 1024, 1024]
            lr  = F.avg_pool2d(t, pool_k).squeeze(0)           # [3, 256, 256]
            images.append(lr)

        # stack and normalize to [-1, 1]
        self.data = torch.stack(images, dim=0) * 2.0 - 1.0    # [N, 3, 256, 256]

        print(f"Preload done!")
        print(f"Dataset shape  : {self.data.shape}")
        print(f"Value range    : [{self.data.min():.3f}, {self.data.max():.3f}]")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]   # [3, lr_size, lr_size]

# CLI test

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',    type=str,
        default='/scratch/bgxp/ezhou1/factor_diffusion_proj/data/all')
    p.add_argument('--pool_k',      type=int, default=4)
    p.add_argument('--orig_size',   type=int, default=1024)
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()

    ds = CelebAHQDataset(
        data_dir=args.data_dir,
        pool_k=args.pool_k,
        orig_size=args.orig_size,
    )

    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)
    batch = next(iter(loader))
    print(f"\nbatch shape : {batch.shape}")   # [B, 3, 256, 256]
    print(f"value range : [{batch.min():.3f}, {batch.max():.3f}]")
    print("Dataset ready.")
