"""model_reaction_1d_dit.py — Conditional DiT for 1D Reaction-Diffusion patch-SVD factors.

Token layout (416 total):
  ┌──────────────────────────────────────────────────────────────────┐
  │  CONDITION (64, not noised)                                      │
  │    alpha_ic (32, 32)  → 32 patch tokens, dim=32  type=2          │
  │    V_hat_ic (640, 32) → 32 rank  tokens, dim=640 type=3          │
  ├──────────────────────────────────────────────────────────────────┤
  │  MAIN (352, noised / denoised)                                   │
  │    alpha (320, 32)    → 320 patch tokens, dim=32  type=0         │
  │    V_hat (640, 32)    →  32 rank tokens, dim=640  type=1         │
  └──────────────────────────────────────────────────────────────────┘

AdaLN conditioning: c = t_embedder(t) + nu_embedder(nu_norm) + rho_embedder(rho_norm).

Flat layouts:
  x_flat    = [alpha.flat (10240) | V_hat.flat (20480)]   dim = 30720
  cond_flat = [alpha_ic.flat (1024) | V_hat_ic.flat (20480)]  dim = 21504
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch
import torch.nn as nn

# Reuse DiT building blocks from the reference repo
sys.path.insert(0, "/u/jlyu5/factor_diffusion")
from video.models.cp_dit import (  # noqa: E402
    TimestepEmbedder, DiTBlock, FinalLayerNorm,
)


# ---------------------------------------------------------------------------
# Tensor-shape constants  (rank=16 patch SVD on 1024x200 trajectories)
# ---------------------------------------------------------------------------
RANK         = 16
PATCH_DIM    = 640

# Patch grid layout for the trajectory (must match save_factors_burgers_1d.py)
N_BLOCK_X = 32   # spatial blocks (1024 / 32)
N_BLOCK_T = 10   # time blocks    (200 / 20)

N_MAIN_PATCH = N_BLOCK_X * N_BLOCK_T   # 320  alpha rows
N_MAIN_RANK  = RANK                    #  32  V_hat columns
N_COND_PATCH = 32                      #  32  alpha_ic rows
N_COND_RANK  = RANK                    #  32  V_hat_ic columns

N_MAIN = N_MAIN_PATCH + N_MAIN_RANK   # 352
N_COND = N_COND_PATCH + N_COND_RANK   #  64
SEQ_LEN = N_COND + N_MAIN              # 416

# flat dimensions
FLAT_ALPHA      = N_MAIN_PATCH * RANK            # 320 * 32 = 10240
FLAT_V_HAT      = PATCH_DIM    * N_MAIN_RANK     # 640 * 32 = 20480
FLAT_MAIN       = FLAT_ALPHA + FLAT_V_HAT        # 30720

FLAT_ALPHA_IC   = N_COND_PATCH * RANK            #  32 * 32 = 1024
FLAT_V_HAT_IC   = PATCH_DIM    * N_COND_RANK     # 640 * 32 = 20480
FLAT_COND       = FLAT_ALPHA_IC + FLAT_V_HAT_IC  # 21504


# ---------------------------------------------------------------------------
# Scalar embedder (continuous input, used for nu)
# ---------------------------------------------------------------------------

class ScalarEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def scalar_embedding(x: torch.Tensor, dim: int, max_period: float = 10000.0):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(x.device)
        args = x[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.scalar_embedding(x, self.frequency_embedding_size))


# ---------------------------------------------------------------------------
# 2D sin-cos positional embedding
# Adapted verbatim from images/our_method/dit_models.py:
#   tuple_to_grid + get_2d_sincos_pos_embed_from_grid + get_1d_sincos_pos_embed_from_grid
# ---------------------------------------------------------------------------

def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Mirrors images/our_method/dit_models.py:get_1d_sincos_pos_embed_from_grid."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega                    # (D/2,)
    pos = pos.reshape(-1)                            # (M,)
    out = np.einsum("m,d->md", pos, omega)           # (M, D/2)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Mirrors images/our_method/dit_models.py:get_2d_sincos_pos_embed_from_grid.

    grid: (2, M).  grid[0]=column coords (x),  grid[1]=row coords (y).
    First half of embed_dim encodes grid[0], second half encodes grid[1].
    """
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _tuple_to_grid(grid_size: tuple[int, int]) -> np.ndarray:
    """Mirrors images/our_method/dit_models.py:tuple_to_grid.

    Returns (2, Gh*Gw); grid[0]=col coord, grid[1]=row coord; row-major flatten.
    """
    Gh, Gw = grid_size
    grid_h = np.arange(Gh, dtype=np.float32)
    grid_w = np.arange(Gw, dtype=np.float32)
    mesh = np.meshgrid(grid_w, grid_h)               # default 'xy' indexing
    grid = np.stack(mesh, axis=0).reshape(2, -1)
    return grid


def make_2d_sincos_pe(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """2D sincos PE for a (Gh, Gw) grid, row-major flatten.  Shape (Gh*Gw, embed_dim).

    Identical convention as images/our_method JointDiT (which calls
    `get_2d_sincos_pos_embed_from_grid(D, tuple_to_grid((Gh, Gw)))`).
    """
    assert embed_dim % 2 == 0
    grid = _tuple_to_grid((grid_h, grid_w))
    pe = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return torch.from_numpy(pe).float()


# ---------------------------------------------------------------------------
# Core DiT
# ---------------------------------------------------------------------------

class ReactionDiT1D(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        depth: int       = 8,
        num_heads: int   = 8,
        mlp_ratio: float = 4.0,
        pos_embed_2d: bool = False,
    ):
        super().__init__()
        H = hidden_size
        self.pos_embed_2d = pos_embed_2d

        # ── Input projections (4 separate, no sharing) ───────────────────
        self.proj_alpha     = nn.Linear(RANK,      H)   # 320 tokens, dim=32  -> H
        self.proj_V_hat     = nn.Linear(PATCH_DIM, H)   #  32 tokens, dim=640 -> H
        self.proj_alpha_ic  = nn.Linear(RANK,      H)   #  32 tokens, dim=32  -> H
        self.proj_V_hat_ic  = nn.Linear(PATCH_DIM, H)   #  32 tokens, dim=640 -> H

        # ── Position embeddings ─────────────────────────────────────────
        # main_patch: optionally 2D sincos over the (N_BLOCK_X, N_BLOCK_T) grid;
        # otherwise 1D learned. Other 3 streams stay 1D learned.
        if pos_embed_2d:
            pe_2d = make_2d_sincos_pe(H, grid_h=N_BLOCK_X, grid_w=N_BLOCK_T)  # (320, H)
            self.register_buffer("pe_main_patch_2d", pe_2d, persistent=False)
            self.pe_main_patch = None
        else:
            self.pe_main_patch = nn.Embedding(N_MAIN_PATCH, H)   # 320
        self.pe_main_rank  = nn.Embedding(N_MAIN_RANK,  H)   #  32
        self.pe_cond_patch = nn.Embedding(N_COND_PATCH, H)   #  32
        self.pe_cond_rank  = nn.Embedding(N_COND_RANK,  H)   #  32

        # ── Token type embeddings (4 types) ──────────────────────────────
        # 0=main_patch (alpha), 1=main_rank (V_hat), 2=cond_patch (alpha_ic), 3=cond_rank (V_hat_ic)
        self.type_embed = nn.Embedding(4, H)

        # ── AdaLN conditioning embedders ─────────────────────────────────
        self.t_embedder   = TimestepEmbedder(H)
        self.nu_embedder  = ScalarEmbedder(H)
        self.rho_embedder = ScalarEmbedder(H)

        # ── Transformer backbone ─────────────────────────────────────────
        self.blocks     = nn.ModuleList(
            [DiTBlock(H, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_norm = FinalLayerNorm(H)

        # ── Output heads (only on main tokens; zero-init) ────────────────
        self.out_alpha = nn.Linear(H, RANK)        # 320 tokens -> dim 32
        self.out_V_hat = nn.Linear(H, PATCH_DIM)   #  32 tokens -> dim 640

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        for head in (self.out_alpha, self.out_V_hat):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        alpha:    torch.Tensor,   # (B, 320, 32)
        V_hat:    torch.Tensor,   # (B, 640, 32)
        alpha_ic: torch.Tensor,   # (B, 32, 32)
        V_hat_ic: torch.Tensor,   # (B, 640, 32)
        t:        torch.Tensor,   # (B,) diffusion timestep
        nu:       torch.Tensor,   # (B,) normalised log-nu
        rho:      torch.Tensor,   # (B,) normalised log-rho
    ):
        device = alpha.device
        B      = alpha.shape[0]

        idx_main_patch = torch.arange(N_MAIN_PATCH, device=device)
        idx_main_rank  = torch.arange(N_MAIN_RANK,  device=device)
        idx_cond_patch = torch.arange(N_COND_PATCH, device=device)
        idx_cond_rank  = torch.arange(N_COND_RANK,  device=device)

        # ── Condition tokens ─────────────────────────────────────────────
        # alpha_ic (B, 32, 32) → 32 patch tokens, dim=32
        h_alpha_ic = (self.proj_alpha_ic(alpha_ic)
                      + self.pe_cond_patch(idx_cond_patch)
                      + self.type_embed.weight[2])               # (B, 32, H)

        # V_hat_ic (B, 640, 32): rank tokens are along last axis → transpose to (B, 32, 640)
        V_hat_ic_t = V_hat_ic.transpose(1, 2)                     # (B, 32, 640)
        h_V_hat_ic = (self.proj_V_hat_ic(V_hat_ic_t)
                      + self.pe_cond_rank(idx_cond_rank)
                      + self.type_embed.weight[3])               # (B, 32, H)

        # ── Main tokens ──────────────────────────────────────────────────
        # alpha (B, 320, 32) → 320 patch tokens, dim=32
        if self.pos_embed_2d:
            pe_alpha = self.pe_main_patch_2d                      # (320, H), buffer
        else:
            pe_alpha = self.pe_main_patch(idx_main_patch)         # (320, H), learned
        h_alpha = (self.proj_alpha(alpha)
                   + pe_alpha
                   + self.type_embed.weight[0])                   # (B, 320, H)

        # V_hat (B, 640, 32) → transpose to (B, 32, 640)
        V_hat_t = V_hat.transpose(1, 2)                            # (B, 32, 640)
        h_V_hat = (self.proj_V_hat(V_hat_t)
                   + self.pe_main_rank(idx_main_rank)
                   + self.type_embed.weight[1])                    # (B, 32, H)

        # ── Concatenate [COND | MAIN] ────────────────────────────────────
        seq = torch.cat([h_alpha_ic, h_V_hat_ic, h_alpha, h_V_hat], dim=1)  # (B, 416, H)

        # ── AdaLN signal: timestep + nu + rho ────────────────────────────
        c = self.t_embedder(t) + self.nu_embedder(nu) + self.rho_embedder(rho)  # (B, H)

        # ── Transformer ──────────────────────────────────────────────────
        for block in self.blocks:
            seq = block(seq, c)
        seq = self.final_norm(seq, c)                               # (B, 416, H)

        # ── Decode main tokens ───────────────────────────────────────────
        s_main      = seq[:, N_COND:]                                # (B, 352, H)
        s_alpha     = s_main[:, :N_MAIN_PATCH]                       # (B, 320, H)
        s_V_hat     = s_main[:, N_MAIN_PATCH:]                       # (B,  32, H)

        pred_alpha  = self.out_alpha(s_alpha)                        # (B, 320, 32)
        pred_V_hat  = self.out_V_hat(s_V_hat)                        # (B,  32, 640)
        pred_V_hat  = pred_V_hat.transpose(1, 2)                     # (B, 640, 32)

        return pred_alpha, pred_V_hat


# ---------------------------------------------------------------------------
# Wrapper: flat ↔ model  (compatible with src.diffusion)
# ---------------------------------------------------------------------------

class Reaction1DWrapper(nn.Module):
    """
    x_flat     : [alpha.flat (10240) | V_hat.flat (20480)]   dim = 30720
    cond_flat  : [alpha_ic.flat (1024) | V_hat_ic.flat (20480)]  dim = 21504
    """

    _split_main  = [FLAT_ALPHA, FLAT_V_HAT]
    _shapes_main = [(N_MAIN_PATCH, RANK), (PATCH_DIM, N_MAIN_RANK)]

    _split_cond  = [FLAT_ALPHA_IC, FLAT_V_HAT_IC]
    _shapes_cond = [(N_COND_PATCH, RANK), (PATCH_DIM, N_COND_RANK)]

    def __init__(self, core: ReactionDiT1D):
        super().__init__()
        self.core = core

    def forward(
        self,
        x_flat:    torch.Tensor,           # (B, 30720)
        t:         torch.Tensor,           # (B,)
        cond_flat: torch.Tensor = None,    # (B, 21504)
        nu:        torch.Tensor = None,    # (B,)
        rho:       torch.Tensor = None,    # (B,)
    ):
        B = x_flat.shape[0]

        c0, c1 = x_flat.split(self._split_main, dim=1)
        alpha = c0.reshape(B, *self._shapes_main[0])    # (B, 320, 32)
        V_hat = c1.reshape(B, *self._shapes_main[1])    # (B, 640, 32)

        if cond_flat is None:
            cond_flat = torch.zeros(B, FLAT_COND, device=x_flat.device,
                                    dtype=x_flat.dtype)
        d0, d1 = cond_flat.split(self._split_cond, dim=1)
        alpha_ic = d0.reshape(B, *self._shapes_cond[0])  # (B, 32, 32)
        V_hat_ic = d1.reshape(B, *self._shapes_cond[1])  # (B, 640, 32)

        if nu is None:
            nu = torch.zeros(B, device=x_flat.device, dtype=x_flat.dtype)
        if rho is None:
            rho = torch.zeros(B, device=x_flat.device, dtype=x_flat.dtype)

        pred_alpha, pred_V_hat = self.core(alpha, V_hat, alpha_ic, V_hat_ic, t, nu, rho)

        return torch.cat([pred_alpha.flatten(1), pred_V_hat.flatten(1)], dim=1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_reaction_1d_dit(cfg: dict) -> Reaction1DWrapper:
    core = ReactionDiT1D(
        hidden_size  = cfg.get("hidden_size", 512),
        depth        = cfg.get("depth",       8),
        num_heads    = cfg.get("num_heads",   8),
        mlp_ratio    = cfg.get("mlp_ratio",   4.0),
        pos_embed_2d = bool(cfg.get("pos_embed_2d", False)),
    )
    return Reaction1DWrapper(core)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B = 2
    wrapper = build_reaction_1d_dit({})
    n_params = sum(p.numel() for p in wrapper.parameters())
    print(f"Model params: {n_params:,}")
    print(f"FLAT_MAIN={FLAT_MAIN}  FLAT_COND={FLAT_COND}  SEQ_LEN={SEQ_LEN}")

    x   = torch.randn(B, FLAT_MAIN)
    cd  = torch.randn(B, FLAT_COND)
    t   = torch.randint(0, 1000, (B,))
    nu  = torch.randn(B)
    rho = torch.randn(B)
    out = wrapper(x, t, cond_flat=cd, nu=nu, rho=rho)
    print(f"x.shape   = {tuple(x.shape)}")
    print(f"out.shape = {tuple(out.shape)}  (should equal x)")
    assert out.shape == x.shape
    print("Forward pass OK.")
