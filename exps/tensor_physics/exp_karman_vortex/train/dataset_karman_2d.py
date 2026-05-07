"""
dataset_karman_2d.py — PyTorch Dataset for 2D Kármán vortex Tucker factors.

Loads Tucker shard files listed in manifest.txt. Each row corresponds to one
clip of the vorticity field.

Factors per row (stored):
  U_T   : (200, r_T)    temporal factor
  U_X   : (128, r_X)    X-spatial factor (r_X = 128, full rank)
  U_Y   : (128, r_Y)    Y-spatial factor
  C     : (r_T, r_X, r_Y) core tensor
  U_ic  : (128, r_ic)   initial-frame U
  Vh_ic : (r_ic, 128)   initial-frame Vh

Following burgers convention, U_X is absorbed into C to form
  G = einsum('xj,tjk->txk', U_X, C)  → (r_T, 128, r_Y)
so only {U_T, U_Y, G} are diffused.

Normalization:
  U_T, U_Y, G, U_ic, Vh_ic : per-tensor z-score
  niu (viscosity), Re      : log then z-score
  cx, cy, r (geometry)     : z-score on raw integers

Conditions used: niu, cx, cy, r, Re.
"""

import os

import torch
from torch.utils.data import Dataset


class KarmanTucker2DDataset(Dataset):
    """
    Parameters
    ----------
    data_dir       : str  path to Tucker data directory (contains manifest.txt)
    test_indices   : list[int] | None
    split          : 'train' | 'test' | 'all'
    device         : torch.device | str
    external_stats : dict | None  If set, skip stats compute/load and use these.
    """

    def __init__(self, data_dir: str, test_indices=None,
                 split: str = 'train', device='cpu',
                 external_stats: dict = None):
        self.device = torch.device(device)
        self.data_dir = data_dir

        manifest = os.path.join(data_dir, 'manifest.txt')
        with open(manifest) as f:
            shard_names = [ln.strip() for ln in f if ln.strip()]

        UT_list, UY_list, G_list = [], [], []
        Uic_list, Vhic_list = [], []
        niu_list, cx_list, cy_list, r_list, Re_list = [], [], [], [], []

        for name in shard_names:
            shard = torch.load(os.path.join(data_dir, name),
                               map_location='cpu', weights_only=False)
            U_T  = shard['U_T'].float()     # (B, 200, r_T)
            U_X  = shard['U_X'].float()     # (B, 128, r_X)
            U_Y  = shard['U_Y'].float()     # (B, 128, r_Y)
            C    = shard['C'].float()       # (B, r_T, r_X, r_Y)
            U_ic = shard['U_ic'].float()    # (B, 128, r_ic)
            Vh_ic= shard['Vh_ic'].float()   # (B, r_ic, 128)

            # G = einsum('bxj,btjk->btxk', U_X, C) → (B, r_T, 128, r_Y)
            G = torch.einsum('bxj,btjk->btxk', U_X, C)

            UT_list.append(U_T)
            UY_list.append(U_Y)
            G_list.append(G)
            Uic_list.append(U_ic)
            Vhic_list.append(Vh_ic)
            niu_list.extend(shard['niu'])
            cx_list.extend(shard['cx'])
            cy_list.extend(shard['cy'])
            r_list.extend(shard['r'])
            Re_list.extend(shard['Re'])

        self.UT_all   = torch.cat(UT_list,   dim=0)
        self.UY_all   = torch.cat(UY_list,   dim=0)
        self.G_all    = torch.cat(G_list,    dim=0)
        self.Uic_all  = torch.cat(Uic_list,  dim=0)
        self.Vhic_all = torch.cat(Vhic_list, dim=0)

        niu_tensor = torch.tensor(niu_list, dtype=torch.float32)
        self.log_niu_all = torch.log(niu_tensor)
        self.cx_all = torch.tensor(cx_list, dtype=torch.float32)
        self.cy_all = torch.tensor(cy_list, dtype=torch.float32)
        self.r_all  = torch.tensor(r_list,  dtype=torch.float32)
        Re_tensor = torch.tensor(Re_list, dtype=torch.float32)
        self.log_Re_all = torch.log(Re_tensor)

        # ── Normalisation stats ─────────────────────────────────────────────
        if external_stats is not None:
            self.stats = {k: (v.detach().clone() if isinstance(v, torch.Tensor)
                              else torch.tensor(v, dtype=torch.float32))
                          for k, v in external_stats.items()}
            print(f'Using external norm stats (keys={list(self.stats.keys())})')
        else:
            stats_path = os.path.join(data_dir, 'norm_stats.pt')
            if os.path.exists(stats_path):
                stats = torch.load(stats_path, map_location='cpu', weights_only=True)
                # Recompute if missing newer keys
                required = {'UT_mean', 'UY_mean', 'G_mean', 'U_ic_mean', 'Vh_ic_mean',
                            'log_niu_mean', 'cx_mean', 'cy_mean', 'r_mean', 'log_Re_mean'}
                if not required.issubset(stats.keys()):
                    stats = self._compute_stats()
                    torch.save(stats, stats_path)
                    print(f'Recomputed norm stats → {stats_path}')
            else:
                stats = self._compute_stats()
                torch.save(stats, stats_path)
                print(f'Saved norm stats → {stats_path}')
            self.stats = {k: torch.tensor(v, dtype=torch.float32) for k, v in stats.items()}

        # ── Move tensors to device ──────────────────────────────────────────
        self.UT_all   = self.UT_all.to(self.device)
        self.UY_all   = self.UY_all.to(self.device)
        self.G_all    = self.G_all.to(self.device)
        self.Uic_all  = self.Uic_all.to(self.device)
        self.Vhic_all = self.Vhic_all.to(self.device)
        self.log_niu_all = self.log_niu_all.to(self.device)
        self.cx_all   = self.cx_all.to(self.device)
        self.cy_all   = self.cy_all.to(self.device)
        self.r_all    = self.r_all.to(self.device)
        self.log_Re_all = self.log_Re_all.to(self.device)
        self.stats    = {k: v.to(self.device) for k, v in self.stats.items()}

        # ── Split indices ───────────────────────────────────────────────────
        N = self.UT_all.shape[0]
        all_idx  = list(range(N))
        test_set = set(test_indices or [])
        if split == 'train':
            self.indices = [i for i in all_idx if i not in test_set]
        elif split == 'test':
            self.indices = [i for i in all_idx if i in test_set]
        else:
            self.indices = all_idx

        print(f'[KarmanTucker2DDataset] split={split}  N={len(self.indices)}'
              f'  UT={tuple(self.UT_all.shape[1:])}  G={tuple(self.G_all.shape[1:])}')

    # ── Stats ──────────────────────────────────────────────────────────────

    def _compute_stats(self) -> dict:
        stats = {}
        for name, tensor in [
            ('UT',    self.UT_all),
            ('UY',    self.UY_all),
            ('G',     self.G_all),
            ('U_ic',  self.Uic_all),
            ('Vh_ic', self.Vhic_all),
        ]:
            stats[f'{name}_mean'] = float(tensor.mean())
            stats[f'{name}_std']  = float(tensor.std().clamp(min=1e-8))
        for name, tensor in [
            ('log_niu', self.log_niu_all),
            ('cx',      self.cx_all),
            ('cy',      self.cy_all),
            ('r',       self.r_all),
            ('log_Re',  self.log_Re_all),
        ]:
            stats[f'{name}_mean'] = float(tensor.mean())
            stats[f'{name}_std']  = float(tensor.std().clamp(min=1e-8))
        return stats

    def _norm(self, x, name):
        return (x - self.stats[f'{name}_mean']) / self.stats[f'{name}_std']

    def denorm(self, x, name):
        mean = self.stats[f'{name}_mean'].to(x.device)
        std  = self.stats[f'{name}_std'].to(x.device)
        return x * std + mean

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]
        return {
            'U_T'   : self._norm(self.UT_all[idx],    'UT'),
            'U_Y'   : self._norm(self.UY_all[idx],    'UY'),
            'G'     : self._norm(self.G_all[idx],     'G'),
            'U_ic'  : self._norm(self.Uic_all[idx],   'U_ic'),
            'Vh_ic' : self._norm(self.Vhic_all[idx],  'Vh_ic'),
            'niu'   : self._norm(self.log_niu_all[idx], 'log_niu'),
            'cx'    : self._norm(self.cx_all[idx],    'cx'),
            'cy'    : self._norm(self.cy_all[idx],    'cy'),
            'r'     : self._norm(self.r_all[idx],     'r'),
            'Re'    : self._norm(self.log_Re_all[idx], 'log_Re'),
            'idx'   : idx,
        }


# ---------------------------------------------------------------------------
# Reconstruction helper
# ---------------------------------------------------------------------------

def reconstruct_video(U_T, U_Y, G):
    """
    Reconstruct (T, X, W) vorticity from absorbed-Tucker factors.
    U_T : (T, r_T)
    U_Y : (W, r_Y)
    G   : (r_T, X, r_Y)
    """
    if isinstance(U_T, torch.Tensor):
        temp  = torch.einsum('txk,wk->txw', G, U_Y)
        video = torch.einsum('ti,ixw->txw', U_T, temp)
    else:
        import numpy as np
        temp  = np.einsum('txk,wk->txw', G, U_Y)
        video = np.einsum('ti,ixw->txw', U_T, temp)
    return video
