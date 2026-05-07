import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse


class BurgersDataset(Dataset):
    def __init__(
        self,
        shard_dir,
        pool_k=8,
        fields=('ux', 'uy'),
        total_timesteps=201,
        samples_per_shard=100,
        shard_prefix='shard_',
    ):
        self.shard_dir = shard_dir
        self.pool_k    = pool_k
        self.fields    = list(fields)

        shard_files = sorted([
            f for f in os.listdir(shard_dir)
            if f.endswith('.pt') and f.startswith(shard_prefix)
        ])

        print(f"Shard dir      : {shard_dir}")
        print(f"Num shards     : {len(shard_files)}")
        print(f"Fields         : {self.fields}")
        print(f"Pool factor k  : {pool_k}  ->  spatial {128//pool_k}x{128//pool_k}")
        print(f"Compression    : {pool_k**2}x")
        print(f"Preload mode   : FULL (all shards into RAM)")

        # full preload
        print(f"Preloading {len(shard_files)} shards into RAM "
              f"(one-time cost, ~5 min)...")
        self.all_data = {}
        for i, shard_file in enumerate(shard_files):
            shard_path = os.path.join(shard_dir, shard_file)
            self.all_data[shard_path] = torch.load(
                shard_path, map_location='cpu', weights_only=False
            )
            if (i + 1) % 10 == 0:
                print(f"  [{i+1:3d}/{len(shard_files)}] shards loaded...")
        print(f"Preload complete! {len(self.all_data)} shards in RAM.")

        # (shard_path, sample_idx, field)
        self.samples = []
        for shard_file in shard_files:
            shard_path = os.path.join(shard_dir, shard_file)
            for sample_idx in range(samples_per_shard):
                for field in self.fields:
                    self.samples.append((shard_path, sample_idx, field))

        print(f"Total samples  : {len(self.samples):,}  "
              f"({len(shard_files)} shards x {samples_per_shard} x {len(self.fields)} fields)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        shard_path, sample_idx, field = self.samples[idx]

        field_data = self.all_data[shard_path][sample_idx][field]  # [201,128,128]

        # condition: t=0 -> [1, 16, 16]
        cond_hr   = field_data[0].unsqueeze(0).unsqueeze(0)        # [1,1,128,128]
        condition = F.avg_pool2d(cond_hr, self.pool_k).squeeze(0)  # [1,16,16]

        # target: t=1..200 -> [200, 16, 16]
        frames_hr = field_data[1:].unsqueeze(1)                    # [200,1,128,128]
        target    = F.avg_pool2d(
            frames_hr.reshape(-1, 1, 128, 128), self.pool_k
        ).squeeze(1)                                               # [200,16,16]

        return {
            'condition': condition,   # [1,   16, 16]
            'target':    target,      # [200, 16, 16]
        }

# CLI test

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--shard_dir', type=str,
        default='/scratch/bgxp/ezhou1/factor_diffusion_proj/data/burgers_2d')
    p.add_argument('--pool_k',            type=int,  default=8)
    p.add_argument('--fields',            nargs='+', default=['ux', 'uy'])
    p.add_argument('--samples_per_shard', type=int,  default=100)
    p.add_argument('--batch_size',        type=int,  default=4)
    p.add_argument('--num_workers',       type=int,  default=4)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()

    ds = BurgersDataset(
        shard_dir=args.shard_dir,
        pool_k=args.pool_k,
        fields=args.fields,
        samples_per_shard=args.samples_per_shard,
    )

    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)
    batch  = next(iter(loader))
    print(f"\ncondition : {batch['condition'].shape}")   # [B, 1,   16, 16]
    print(f"target    : {batch['target'].shape}")        # [B, 200, 16, 16]
    print(f"cond range: [{batch['condition'].min():.4f}, {batch['condition'].max():.4f}]")
    print(f"tgt  range: [{batch['target'].min():.4f}, {batch['target'].max():.4f}]")
    print("Dataset ready.")
