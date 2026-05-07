import glob
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def _list_shards(root, prefix):
    return sorted(glob.glob(os.path.join(root, f'{prefix}*.pt')))


class KarmanShardedDataset(Dataset):
    """Lazy loader for sharded Karman .pt files.

    Each __getitem__(idx) returns:
        clip:       FloatTensor of shape (T, 1, H, W)   — vorticity, channel-first
        global_idx: int                                 — index into tucker_core
    """

    def __init__(self, root, split='train', T=201, max_clips=None,
                 clips_per_shard=50):
        if split == 'train':
            shard_root = root
            prefix = 'shard_'
        elif split == 'test':
            shard_root = os.path.join(root, 'test_data')
            prefix = 'test_shard_'
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        shards = _list_shards(shard_root, prefix)
        if not shards:
            raise FileNotFoundError(
                f"No shards found under {shard_root} with prefix {prefix!r}")
        self.shards = shards
        self.T = int(T)
        self.clips_per_shard = int(clips_per_shard)

        total = self.clips_per_shard * len(self.shards)
        self.N = total if max_clips is None else min(int(max_clips), total)

        # Per-worker shard cache. PyTorch DataLoader copies this dataset object
        # to each worker process, so each worker keeps its own cached shard.
        self._cached_idx = -1
        self._cached_shard = None

    def __len__(self):
        return self.N

    def _get_shard(self, shard_idx):
        if shard_idx != self._cached_idx:
            self._cached_shard = torch.load(
                self.shards[shard_idx], map_location='cpu', weights_only=False)
            self._cached_idx = shard_idx
        return self._cached_shard

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.N:
            raise IndexError(idx)
        shard_idx, local_idx = divmod(idx, self.clips_per_shard)
        shard = self._get_shard(shard_idx)
        sample = shard[local_idx]
        clip = sample['vor']
        if isinstance(clip, np.ndarray):
            clip = torch.from_numpy(clip)
        clip = clip.float()
        if clip.shape[0] != self.T:
            clip = clip[:self.T]
        # Add channel dim: (T, H, W) -> (T, 1, H, W)
        clip = clip.unsqueeze(1).contiguous()
        return clip, idx


class ShardSequentialSampler(Sampler):
    """Sampler that yields indices grouped by shard.

    With shard_size=50 and batch_size=32, each shard yields one batch of 32 and
    one batch of 18 (or drops the partial). Across an epoch every clip in a
    shard is consumed before moving to the next shard, so each worker only
    reloads a shard when the sampler advances to a new one.

    With shuffle=True we randomise both the shard order and the within-shard
    order — only intra-shard permutation requires no extra disk IO.
    """

    def __init__(self, num_shards, clips_per_shard, batch_size,
                 shuffle=True, drop_last=True, seed=0, total_clips=None):
        self.num_shards = num_shards
        self.clips_per_shard = clips_per_shard
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        # If max_clips < num_shards*clips_per_shard, only emit indices < total_clips.
        self.total_clips = (num_shards * clips_per_shard if total_clips is None
                            else int(total_clips))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.shuffle:
            shard_order = torch.randperm(self.num_shards, generator=g).tolist()
        else:
            shard_order = list(range(self.num_shards))
        for s in shard_order:
            if self.shuffle:
                local = torch.randperm(self.clips_per_shard, generator=g).tolist()
            else:
                local = list(range(self.clips_per_shard))
            global_indices = [s * self.clips_per_shard + i for i in local
                              if s * self.clips_per_shard + i < self.total_clips]
            for i in range(0, len(global_indices), self.batch_size):
                batch = global_indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        # Approximate: actual count varies if total_clips truncates a shard.
        full = (self.total_clips // self.batch_size)
        if not self.drop_last and self.total_clips % self.batch_size:
            full += 1
        return full


def build_loader(dataset, batch_size, num_workers=4, shuffle=True,
                 drop_last=True, seed=0):
    sampler = ShardSequentialSampler(
        num_shards=len(dataset.shards),
        clips_per_shard=dataset.clips_per_shard,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
        total_clips=len(dataset),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def get_grid_coords(H=128, W=128, device='cpu'):
    """Returns the (u, v, w) coordinate tensors that the SDIFT FTM expects.

    For 2D Karman with (channel=1, H, W) layout:
      u_ind_uni: shape (1,)    -- single channel, value 1.0
      v_ind_uni: shape (H,)    -- normalised row coords in [0, 1]
      w_ind_uni: shape (W,)    -- normalised col coords in [0, 1]
    """
    u = torch.ones(1, device=device)
    v = torch.linspace(0., 1., H, device=device)
    w = torch.linspace(0., 1., W, device=device)
    return u, v, w
