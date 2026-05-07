import torch
import numpy as np


def total_variation_loss_1d(X, weight: float = 1e-7):
    """TV loss across the time axis of a (B, T, R) core tensor."""
    diff = X[:, 1:, :] - X[:, :-1, :]
    return weight * torch.sum(torch.norm(diff, p="fro", dim=1))


def get_gp_covariance(t, gp_gamma: float = 50.0):
    """Squared-exponential kernel over the time axis.

    t: (B, T, 1).  Returns (B, T, T).
    """
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2], device=t.device, dtype=t.dtype) * 1e-5
    return torch.exp(-torch.square(s) * gp_gamma) + diag


class BurgersPTDataset(torch.utils.data.Dataset):
    """Memory-mapped reader for /scratch/bkx8728/burgers_1d/*.pt files.

    Each item: (u_tx[T,X], nu_scalar, idx).
    """

    def __init__(self, pt_path: str):
        self.pt_path = pt_path
        d = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)
        self.N, self.T, self.X = d["tensor"].shape
        # Eager copies of metadata (small).
        self.nu = d["nu"].float().clone()
        self.x_coord = d["x_coord"].float().clone()
        self.t_coord = d["t_coord"].float().clone()
        self._mm = None  # lazy per-worker handle

    def __len__(self):
        return self.N

    def _ensure_mm(self):
        if self._mm is None:
            self._mm = torch.load(
                self.pt_path, map_location="cpu", weights_only=False, mmap=True
            )

    def __getitem__(self, idx):
        self._ensure_mm()
        u = self._mm["tensor"][idx].clone()
        return u, self.nu[idx], idx


class Reaction1DPTDataset(torch.utils.data.Dataset):
    """Memory-mapped reader for /scratch/bkx8728/reaction_1d/*.pt files.

    Reaction-diffusion data has two scalar parameters per sample (nu, rho).
    Each item: (u_tx[T,X], nu_scalar, rho_scalar, idx).
    """

    def __init__(self, pt_path: str):
        self.pt_path = pt_path
        d = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)
        self.N, self.T, self.X = d["tensor"].shape
        self.nu = d["nu"].float().clone()
        self.rho = d["rho"].float().clone()
        self.x_coord = d["x_coord"].float().clone()
        self.t_coord = d["t_coord"].float().clone()
        self._mm = None

    def __len__(self):
        return self.N

    def _ensure_mm(self):
        if self._mm is None:
            self._mm = torch.load(
                self.pt_path, map_location="cpu", weights_only=False, mmap=True
            )

    def __getitem__(self, idx):
        self._ensure_mm()
        u = self._mm["tensor"][idx].clone()
        return u, self.nu[idx], self.rho[idx], idx
