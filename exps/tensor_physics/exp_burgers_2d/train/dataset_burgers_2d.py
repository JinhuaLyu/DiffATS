"""
dataset_burgers_2d.py — PyTorch Dataset for 2D Burgers Tucker factors.

Loads Tucker shard files listed in manifest.txt.  Each row in a shard
represents one PDE component (ux or uy) for one physical sample.

Factors per row:
  U_1   : (200, r_T)   temporal factor
  U_2   : (128, r_H)   H-spatial factor
  U_3   : (128, r_W)   W-spatial factor
  C     : (r_T, r_H, r_W) core tensor
  U_ic  : (128, r_ic)  initial-frame U factor
  Vh_ic : (r_ic, 128)  initial-frame Vh factor
  nu    : float        scalar viscosity (shared between ux/uy of same sample)

G = einsum('hj,tjk->thk', U_2, C) is pre-computed and stored instead of
keeping U_2 and C separately, matching the reference implementation.

Normalization:
  All factor tensors: z-score (per-tensor mean/std across the dataset)
  nu: log(nu) then z-score

Norm stats are computed once and cached to norm_stats.pt in data_dir.
All tensors are preloaded onto the target device at __init__ time.
"""

import os

import torch
from torch.utils.data import Dataset


class BurgersTucker2DDataset(Dataset):
    """
    Parameters
    ----------
    data_dir     : str           path to Tucker data directory (contains manifest.txt)
    test_indices : list[int] | None
    split        : 'train' | 'test' | 'all'
    device       : torch.device | str
    """

    def __init__(self, data_dir: str, test_indices=None,
                 split: str = 'train', device='cpu',
                 external_stats: dict = None):
        self.device = torch.device(device)
        self.data_dir = data_dir

        # ── Load all shards listed in manifest ──────────────────────────────
        manifest = os.path.join(data_dir, 'manifest.txt')
        with open(manifest) as f:
            shard_names = [ln.strip() for ln in f if ln.strip()]

        U1_list, U3_list, G_list = [], [], []
        Uic_list, Vhic_list, nu_list, cd_list = [], [], [], []

        for name in shard_names:
            shard_path = os.path.join(data_dir, name)
            shard = torch.load(shard_path, map_location='cpu', weights_only=False)

            U_1  = shard['U_1'].float()    # (B, 200, r_T)
            U_2  = shard['U_2'].float()    # (B, 128, r_H)
            U_3  = shard['U_3'].float()    # (B, 128, r_W)
            C    = shard['C'].float()      # (B, r_T, r_H, r_W)
            U_ic = shard['U_ic'].float()   # (B, 128, r_ic)
            Vh_ic= shard['Vh_ic'].float()  # (B, r_ic, 128)
            nu_vals = shard['nu']          # list[float]
            cd_vals = shard['cd']          # list[float]

            # G = einsum('bhj,btjk->bthk', U_2, C)  → (B, r_T, 128, r_W)
            G = torch.einsum('bhj,btjk->bthk', U_2, C)

            U1_list.append(U_1)
            U3_list.append(U_3)
            G_list.append(G)
            Uic_list.append(U_ic)
            Vhic_list.append(Vh_ic)
            nu_list.extend(nu_vals)
            cd_list.extend(cd_vals)

        self.U1_all   = torch.cat(U1_list,   dim=0)   # (N, 200, r_T)
        self.U3_all   = torch.cat(U3_list,   dim=0)   # (N, 128, r_W)
        self.G_all    = torch.cat(G_list,    dim=0)   # (N, r_T, 128, r_W)
        self.Uic_all  = torch.cat(Uic_list,  dim=0)   # (N, 128, r_ic)
        self.Vhic_all = torch.cat(Vhic_list, dim=0)   # (N, r_ic, 128)
        nu_tensor     = torch.tensor(nu_list, dtype=torch.float32)  # (N,)
        cd_tensor     = torch.tensor(cd_list, dtype=torch.float32)  # (N,)

        # ── log-transform nu; cd is already on a linear scale ────────────────
        self.log_nu_all = torch.log(nu_tensor)   # (N,)
        self.cd_all     = cd_tensor              # (N,)  range ~ [-1, -0.1]

        # ── Compute or load norm stats ────────────────────────────────────────
        if external_stats is not None:
            # Reuse stats passed from another dataset (e.g. train stats for test)
            self.stats = {k: (v.detach().clone() if isinstance(v, torch.Tensor)
                              else torch.tensor(v, dtype=torch.float32))
                          for k, v in external_stats.items()}
            print(f'Using external norm stats (keys={list(self.stats.keys())})')
        else:
            stats_path = os.path.join(data_dir, 'norm_stats.pt')
            if os.path.exists(stats_path):
                stats = torch.load(stats_path, map_location='cpu', weights_only=True)
                if 'cd_mean' not in stats:   # recompute if cd was not previously saved
                    stats = self._compute_stats()
                    torch.save(stats, stats_path)
                    print(f'Recomputed norm stats (cd added) → {stats_path}')
            else:
                stats = self._compute_stats()
                torch.save(stats, stats_path)
                print(f'Saved norm stats → {stats_path}')
            self.stats = {k: torch.tensor(v, dtype=torch.float32) for k, v in stats.items()}

        # ── Move all data to target device ────────────────────────────────────
        self.U1_all   = self.U1_all.to(self.device)
        self.U3_all   = self.U3_all.to(self.device)
        self.G_all    = self.G_all.to(self.device)
        self.Uic_all  = self.Uic_all.to(self.device)
        self.Vhic_all = self.Vhic_all.to(self.device)
        self.log_nu_all = self.log_nu_all.to(self.device)
        self.cd_all     = self.cd_all.to(self.device)
        self.stats    = {k: v.to(self.device) for k, v in self.stats.items()}

        # ── Build index list based on split ───────────────────────────────────
        N = self.U1_all.shape[0]
        all_idx  = list(range(N))
        test_set = set(test_indices or [])

        if split == 'train':
            self.indices = [i for i in all_idx if i not in test_set]
        elif split == 'test':
            self.indices = [i for i in all_idx if i in test_set]
        else:
            self.indices = all_idx

        print(f'[BurgersTucker2DDataset] split={split}  N={len(self.indices)}'
              f'  U1={tuple(self.U1_all.shape[1:])}  G={tuple(self.G_all.shape[1:])}')

    # ── Statistics ─────────────────────────────────────────────────────────────

    def _compute_stats(self) -> dict:
        stats = {}
        for name, tensor in [
            ('U1',     self.U1_all),
            ('U3',     self.U3_all),
            ('G',      self.G_all),
            ('U_ic',   self.Uic_all),
            ('Vh_ic',  self.Vhic_all),
        ]:
            stats[f'{name}_mean'] = float(tensor.mean())
            stats[f'{name}_std']  = float(tensor.std().clamp(min=1e-8))
        stats['log_nu_mean'] = float(self.log_nu_all.mean())
        stats['log_nu_std']  = float(self.log_nu_all.std().clamp(min=1e-8))
        stats['cd_mean']     = float(self.cd_all.mean())
        stats['cd_std']      = float(self.cd_all.std().clamp(min=1e-8))
        return stats

    # ── Normalization ───────────────────────────────────────────────────────────

    def _norm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return (x - self.stats[f'{name}_mean']) / self.stats[f'{name}_std']

    def denorm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        mean = self.stats[f'{name}_mean'].to(x.device)
        std  = self.stats[f'{name}_std'].to(x.device)
        return x * std + mean

    # ── Dataset interface ───────────────────────────────────────────────────────

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]
        return {
            'U1'    : self._norm(self.U1_all[idx],    'U1'),
            'U3'    : self._norm(self.U3_all[idx],    'U3'),
            'G'     : self._norm(self.G_all[idx],     'G'),
            'U_ic'  : self._norm(self.Uic_all[idx],   'U_ic'),
            'Vh_ic' : self._norm(self.Vhic_all[idx],  'Vh_ic'),
            'nu'    : self._norm(self.log_nu_all[idx], 'log_nu'),
            'cd'    : self._norm(self.cd_all[idx],     'cd'),
            'idx'   : idx,
        }


# ---------------------------------------------------------------------------
# Reconstruction helper
# ---------------------------------------------------------------------------

def reconstruct_video(U1, U3, G):
    """
    Reconstruct (T, H, W) tensor from Tucker factors.

    U1 : (T, r_T)
    U3 : (W, r_W)          W-spatial factor
    G  : (r_T, H, r_W)     merged factor  G = C ×_2 U_2
    """
    if isinstance(U1, torch.Tensor):
        # G(r_T,H,r_W), U3(W,r_W) → temp(r_T,H,W)
        temp  = torch.einsum('thk,wk->thw', G, U3)
        # U1(T,r_T), temp(r_T,H,W) → video(T,H,W)
        video = torch.einsum('ti,ihw->thw', U1, temp)
    else:
        import numpy as np
        temp  = np.einsum('thk,wk->thw', G, U3)
        video = np.einsum('ti,ihw->thw', U1, temp)
    return video


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else \
        'data/tucker_burgers_rT5_rH20_rW20'
    ds = BurgersTucker2DDataset(data_dir, split='all')
    sample = ds[0]
    print('Sample keys:', list(sample.keys()))
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: shape={tuple(v.shape)}  min={v.min():.3f}  max={v.max():.3f}')
        else:
            print(f'  {k}: {v}')
    # test reconstruction
    U1  = ds.denorm(sample['U1'],    'U1').cpu()
    U3  = ds.denorm(sample['U3'],    'U3').cpu()
    G   = ds.denorm(sample['G'],     'G').cpu()
    vid = reconstruct_video(U1, U3, G)
    print(f'Reconstructed video: {tuple(vid.shape)}')
