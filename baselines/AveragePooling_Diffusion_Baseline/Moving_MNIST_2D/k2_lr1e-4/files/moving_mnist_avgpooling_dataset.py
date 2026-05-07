import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse


class MovingMNISTDataset(Dataset):

    def __init__(
        self,
        data_path,
        pool_k=2,
    ):
        self.pool_k = pool_k

        print(f"Loading data from: {data_path}")
        raw = torch.load(data_path, map_location='cpu', weights_only=False)
        # raw: [20, 20000, 64, 64] uint8

        print(f"Raw shape  : {raw.shape}  dtype={raw.dtype}")
        print(f"Pool factor: {pool_k}  ->  spatial {64//pool_k}x{64//pool_k}")
        print(f"Compression: {pool_k**2}x")

        # rearrange to [N, T, H, W] and normalize to [-1, 1]
        # raw: [T, N, H, W] -> [N, T, H, W]
        data = raw.permute(1, 0, 2, 3).float() / 127.5 - 1.0  # [20000, 20, 64, 64]

        # avg pool: [N*T, 1, H, W] -> [N*T, 1, H/k, W/k]
        N, T, H, W = data.shape
        data_flat = data.reshape(N * T, 1, H, W)
        data_lr   = F.avg_pool2d(data_flat, pool_k)             # [N*T, 1, 32, 32]
        self.data = data_lr.reshape(N, T, H//pool_k, W//pool_k) # [N, T, 32, 32]

        print(f"Dataset shape: {self.data.shape}")
        print(f"Value range  : [{self.data.min():.3f}, {self.data.max():.3f}]")
        print(f"Total samples: {len(self.data):,}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]   # [20, 32, 32]


# CLI test

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str,
        default='/scratch/bgxp/ezhou1/factor_diffusion_proj/data/moving_mnist/moving_mnist_20k_2slow.pt')
    p.add_argument('--pool_k',      type=int, default=2)
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()

    ds = MovingMNISTDataset(
        data_path=args.data_path,
        pool_k=args.pool_k,
    )

    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)
    batch = next(iter(loader))
    print(f"\nbatch shape : {batch.shape}")   # [B, 20, 32, 32]
    print(f"value range : [{batch.min():.3f}, {batch.max():.3f}]")
    print("Dataset ready.")
