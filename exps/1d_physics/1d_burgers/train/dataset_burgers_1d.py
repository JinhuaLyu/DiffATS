"""dataset_burgers_1d.py — PyTorch Dataset for 1D Burgers patch-SVD factors.

Loads a SINGLE .pt produced by `data_tucker/save_factors_burgers_1d.py`:

    {
        "alpha"    : (N, 320, 32),    # main: 320 patch tokens, dim 32
        "V_hat"    : (N, 640, 32),    # main:  32 rank tokens,  dim 640
        "alpha_ic" : (N,  32, 32),    # cond:  32 patch tokens, dim 32
        "V_hat_ic" : (N, 640, 32),    # cond:  32 rank tokens,  dim 640
        "nu"       : (N,)             # scalar viscosity per sample
        ...
    }

Normalization (scale-only, mirrors images/our_method/compute_alpha_stats_refimg.py):
    alpha    : per-rank std (32,)         x_norm = x / std[None, None, :]   ; denorm = x * std
    V_hat    : scalar std                 x_norm = x / std                  ; denorm = x * std
    alpha_ic : per-rank std (32,)         (same)
    V_hat_ic : scalar std                 (same)

    nu       : log(nu) z-score (mean + std).  Used as scalar AdaLN conditioning;
               not bilinear, so full z-score is fine here.

Why scale-only for the four factor tensors?
    Reconstruction is bilinear (alpha @ V_hat.T).  Subtracting the mean makes
    the round-trip introduce cross terms (mu_a @ V_norm.T, alpha_norm @ mu_v.T)
    that the model never sees during training, contaminating samples.  Plain
    scaling is invariant: (alpha/sigma_a) @ (V/sigma_v).T = (alpha @ V.T) / (sigma_a sigma_v).

Stats are cached to <stats_dir>/norm_stats.pt; pass `external_stats` to reuse
train-set stats for the test set.

Tensors stay on CPU (`device` arg only affects stats device for fast denorm
during sampling/visualization); each training batch is moved to GPU in the
training loop, allowing num_workers > 0.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Tensor-shape constants (must match save_factors_burgers_1d.py)
# ---------------------------------------------------------------------------
N_MAIN_PATCH  = 320
N_MAIN_RANK   = 32
N_COND_PATCH  = 32
N_COND_RANK   = 32
PATCH_DIM     = 640
RANK          = 32


class BurgersFactor1DDataset(Dataset):
    """
    Parameters
    ----------
    data_path     : str   path to a single .pt file with the factors above
    stats_dir     : str | None  directory to cache norm_stats.pt (defaults to
                                parent directory of data_path)
    test_indices  : list[int] | None  indices reserved for test split
    split         : 'train' | 'test' | 'all'
    device        : torch.device | str  device for stats (data tensors stay on CPU)
    external_stats: dict | None  reuse train stats for the test dataset
    """

    def __init__(
        self,
        data_path: str,
        stats_dir: str | None = None,
        test_indices: list[int] | None = None,
        split: str = "all",
        device: str | torch.device = "cpu",
        external_stats: dict | None = None,
    ):
        self.device = torch.device(device)
        self.data_path = data_path
        self.stats_dir = Path(stats_dir) if stats_dir else Path(data_path).parent

        # ── Load full payload onto CPU ────────────────────────────────────
        payload = torch.load(data_path, map_location="cpu", weights_only=False)
        self.alpha_all     = payload["alpha"].float()      # (N, 320, 32)
        self.V_hat_all     = payload["V_hat"].float()      # (N, 640, 32)
        self.alpha_ic_all  = payload["alpha_ic"].float()   # (N, 32, 32)
        self.V_hat_ic_all  = payload["V_hat_ic"].float()   # (N, 640, 32)
        nu = payload["nu"]
        nu_tensor = nu.float() if isinstance(nu, torch.Tensor) else torch.tensor(nu, dtype=torch.float32)
        self.log_nu_all = torch.log(nu_tensor)              # (N,)

        N = self.alpha_all.shape[0]
        assert self.alpha_all.shape    == (N, N_MAIN_PATCH, RANK)
        assert self.V_hat_all.shape    == (N, PATCH_DIM, RANK)
        assert self.alpha_ic_all.shape == (N, N_COND_PATCH, RANK)
        assert self.V_hat_ic_all.shape == (N, PATCH_DIM, RANK)
        assert self.log_nu_all.shape   == (N,)

        # ── Norm stats: load / compute / passed-in ────────────────────────
        if external_stats is not None:
            self.stats = {
                k: (v.detach().clone() if isinstance(v, torch.Tensor)
                    else torch.tensor(v, dtype=torch.float32))
                for k, v in external_stats.items()
            }
            print(f"[norm] reusing external stats keys={list(self.stats.keys())}")
        else:
            self.stats_dir.mkdir(parents=True, exist_ok=True)
            stats_path = self.stats_dir / "norm_stats.pt"
            if stats_path.exists():
                stats_dict = torch.load(stats_path, map_location="cpu", weights_only=True)
                print(f"[norm] loaded stats from {stats_path}")
            else:
                stats_dict = self._compute_stats()
                torch.save(stats_dict, stats_path)
                print(f"[norm] saved stats -> {stats_path}")
            self.stats = {
                k: (v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32))
                for k, v in stats_dict.items()
            }

        # Stats stay on CPU (so DataLoader workers can access them via fork);
        # `denorm()` moves them to x.device on the fly.

        # ── Index list per split ──────────────────────────────────────────
        all_idx = list(range(N))
        test_set = set(test_indices or [])
        if split == "train":
            self.indices = [i for i in all_idx if i not in test_set]
        elif split == "test":
            self.indices = [i for i in all_idx if i in test_set]
        else:
            self.indices = all_idx

        print(
            f"[BurgersFactor1DDataset] split={split}  N={len(self.indices)}"
            f"  alpha={tuple(self.alpha_all.shape[1:])}"
            f"  V_hat={tuple(self.V_hat_all.shape[1:])}"
            f"  alpha_std={'per-rank' if self.stats['alpha_std'].numel() > 1 else 'scalar'}"
            f"  V_hat_std={'per-rank' if self.stats['V_hat_std'].numel() > 1 else 'scalar'}"
        )

    # ── Stats ─────────────────────────────────────────────────────────────
    def _compute_stats(self) -> dict:
        s = {}
        # Per-rank std for alpha-side tensors (shape (RANK,))
        s["alpha_std"]     = self.alpha_all.std(dim=(0, 1))      # (32,)
        s["alpha_ic_std"]  = self.alpha_ic_all.std(dim=(0, 1))   # (32,)
        # Scalar std for V_hat-side tensors
        s["V_hat_std"]     = self.V_hat_all.std()                # ()
        s["V_hat_ic_std"]  = self.V_hat_ic_all.std()             # ()
        # log(nu) full z-score (scalar); used in AdaLN, not bilinear
        s["log_nu_mean"]   = self.log_nu_all.mean()
        s["log_nu_std"]    = self.log_nu_all.std().clamp(min=1e-8)
        return s

    # ── Norm helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _scale_only(x: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # Broadcast: scalar std -> trivial; (R,) std -> aligned with last dim of x
        return x / std

    @staticmethod
    def _scale_only_inv(x: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std

    def denorm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Inverse of __getitem__'s normalization for a given factor tensor or nu."""
        if name == "log_nu":
            mean = self.stats["log_nu_mean"].to(x.device)
            std  = self.stats["log_nu_std"].to(x.device)
            return x * std + mean
        std = self.stats[f"{name}_std"].to(x.device)
        return self._scale_only_inv(x, std)

    # ── Dataset interface ─────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        idx = self.indices[item]
        return {
            "alpha":    self._scale_only(self.alpha_all[idx],    self.stats["alpha_std"]),
            "V_hat":    self._scale_only(self.V_hat_all[idx],    self.stats["V_hat_std"]),
            "alpha_ic": self._scale_only(self.alpha_ic_all[idx], self.stats["alpha_ic_std"]),
            "V_hat_ic": self._scale_only(self.V_hat_ic_all[idx], self.stats["V_hat_ic_std"]),
            "nu":       (self.log_nu_all[idx] - self.stats["log_nu_mean"])
                        / self.stats["log_nu_std"],
            "idx":      idx,
        }


# ---------------------------------------------------------------------------
# Reconstruction helpers (operate on de-normalised tensors)
# ---------------------------------------------------------------------------

PATCH_X = 32
PATCH_T = 20
NX      = 1024
T_TRAJ  = 200
N_BLOCK_X = NX // PATCH_X       # 32
N_BLOCK_T = T_TRAJ // PATCH_T   # 10


def reconstruct_traj(alpha: torch.Tensor, V_hat: torch.Tensor) -> torch.Tensor:
    """alpha (..., 320, 32), V_hat (..., 640, 32) -> (..., 1024, 200).

    Inverse of `patchify_traj` in save_factors_burgers_1d.py:
      A (320, 640) -> (32, 10, 32, 20) -> permute (32, 32, 10, 20) -> (1024, 200)
    """
    A = alpha @ V_hat.transpose(-1, -2)
    lead = A.shape[:-2]
    A = A.reshape(*lead, N_BLOCK_X, N_BLOCK_T, PATCH_X, PATCH_T)
    perm = list(range(len(lead))) + [
        len(lead) + 0, len(lead) + 2, len(lead) + 1, len(lead) + 3,
    ]
    A = A.permute(*perm).contiguous()
    return A.reshape(*lead, NX, T_TRAJ)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else \
        "${DATA_ROOT}/tucker_factors/burgers_1d/burgers_1d_train.pt"
    ds = BurgersFactor1DDataset(p, split="all")
    s = ds[0]
    for k, v in s.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:10s} shape={tuple(v.shape)}  mean={v.mean():.3f}  std={v.std():.3f}")
        else:
            print(f"  {k:10s} = {v}")
    # reconstruction sanity (round-trip should be exact)
    alpha = ds.denorm(s["alpha"], "alpha")
    V_hat = ds.denorm(s["V_hat"], "V_hat")
    traj = reconstruct_traj(alpha, V_hat)
    print(f"reconstruct_traj output: {tuple(traj.shape)}  norm={traj.norm():.3f}")

    print("\n[stats]")
    for k, v in ds.stats.items():
        print(f"  {k}: shape={tuple(v.shape)}  mean={float(v.mean()):.4f}  "
              f"min={float(v.min()):.4f}  max={float(v.max()):.4f}")
