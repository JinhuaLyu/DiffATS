import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse


class KarmanDataset(Dataset):
    def __init__(
        self,
        shard_dir,
        pool_k=8,
        total_timesteps=201,
        samples_per_shard=50,
        shard_prefix='shard_',
        preload_all=True,
        max_cached_shards=4,
    ):
        self.shard_dir     = shard_dir
        self.pool_k        = pool_k
        self.preload_all   = preload_all

        shard_files = sorted([
            f for f in os.listdir(shard_dir)
            if f.endswith('.pt') and f.startswith(shard_prefix)
        ])

        print(f"Shard dir      : {shard_dir}")
        print(f"Num shards     : {len(shard_files)}")
        print(f"Field          : vor (vorticity)")
        print(f"Pool factor k  : {pool_k}  →  spatial {128//pool_k}x{128//pool_k}")
        print(f"Compression    : {pool_k**2}x")
        print(f"Preload all    : {preload_all}")

        self.samples = []

        if preload_all:
            # full preload，worker share thru fork
            print("Preloading all shards into RAM (one-time, ~5min)...")
            self.all_data = {}
            for i, shard_file in enumerate(shard_files):
                shard_path = os.path.join(shard_dir, shard_file)
                self.all_data[shard_path] = torch.load(
                    shard_path, map_location='cpu', weights_only=False
                )
                if (i + 1) % 20 == 0:
                    print(f"  Loaded {i+1}/{len(shard_files)} shards...")
            print(f"Preload done! {len(self.all_data)} shards in RAM.")
            self.shard_cache = self.all_data
        else:
            self.shard_cache = {}
            self.max_cached_shards = max_cached_shards

        for shard_file in shard_files:
            shard_path = os.path.join(shard_dir, shard_file)
            for sample_idx in range(samples_per_shard):
                self.samples.append((shard_path, sample_idx))

        print(f"Total samples  : {len(self.samples):,}")

    def _load_shard(self, shard_path):
        if shard_path not in self.shard_cache:
            if len(self.shard_cache) >= self.max_cached_shards:
                oldest = next(iter(self.shard_cache))
                del self.shard_cache[oldest]
            self.shard_cache[shard_path] = torch.load(
                shard_path, map_location='cpu', weights_only=False
            )
        return self.shard_cache[shard_path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        shard_path, sample_idx = self.samples[idx]

        if self.preload_all:
            shard = self.all_data[shard_path]
        else:
            shard = self._load_shard(shard_path)

        field_data = shard[sample_idx]['vor']          # [201, 128, 128]

        cond_hr   = field_data[0].unsqueeze(0).unsqueeze(0)
        condition = F.avg_pool2d(cond_hr, self.pool_k).squeeze(0)  # [1,16,16]

        frames_hr = field_data[1:].unsqueeze(1)        # [200,1,128,128]
        target    = F.avg_pool2d(
            frames_hr.reshape(-1, 1, 128, 128), self.pool_k
        ).squeeze(1)                                   # [200,16,16]

        return {
            'condition': condition,
            'target':    target,
        }
