"""
model_burgers_2d_dit.py — Conditional DiT for 2D Burgers Tucker factor diffusion.

Tucker rank = [r_T=5, r_H=20, r_W=20] on (T=200, H=128, W=128).
Condition: initial frame (U_ic, Vh_ic) + scalar viscosity nu.

Factor shapes (after merging U_2 into G):
  U_1   : (200, r_T)       temporal factor
  U_3   : (128, r_W)       W-spatial factor
  G     : (r_T, 128, r_W)  merged factor  G = C ×_2 U_2

Token layout (813 tokens total):
  ┌────────────────────────────────────────────────────────────────────┐
  │  CONDITION (148, not noised)                                       │
  │    U_ic  (128, 20) → 128 spatial tokens, dim=20   type=3          │
  │    Vh_ic (20, 128) →  20 rank tokens,   dim=128   type=4          │
  ├────────────────────────────────────────────────────────────────────┤
  │  MAIN (665, noised / denoised)                                     │
  │    U_1 (200,5) transposed → 5 rank tokens,   dim=200   type=0     │
  │    U_3 (128,20) transposed→ 20 rank tokens,  dim=128   type=1     │
  │    G   (5,128,20) reshaped→ 640 spatial tokens, dim=20 type=2     │
  └────────────────────────────────────────────────────────────────────┘

nu conditioning: scalar embedding added to the AdaLN condition vector c,
alongside the diffusion timestep embedding:
    c = t_embedder(t) + nu_embedder(nu_norm)

x_flat layout  : [U1.flat (1000) | U3.flat (2560) | G.flat (12800)]  dim=16360
cond_flat layout: [Uic.flat (2560) | Vhic.flat (2560)]               dim=5120
"""

import math
import os
import sys

import torch
import torch.nn as nn

# Reuse DiT building blocks from the reference repo
sys.path.insert(0, '${REPO_ROOT}')
from video.models.cp_dit import (
    Mlp, modulate,
    TimestepEmbedder, SDPAAttention, DiTBlock, FinalLayerNorm,
)

# ---------------------------------------------------------------------------
# Dimension constants  (rank=[5,20,20], r_ic=20, T=200, H=W=128)
# ---------------------------------------------------------------------------
R_T, R_H, R_W = 5, 20, 20   # Tucker rank
R_IC  = 20                   # IC SVD rank (= r_H = r_W by construction)
T_DIM = 200                  # temporal length
H_DIM = 128                  # spatial H = W

# main tokens
N_U1   = R_T              # 5
N_U3   = R_W              # 20
N_G    = R_T * H_DIM      # 640
N_MAIN = N_U1 + N_U3 + N_G   # 665

# condition tokens
N_Uic  = H_DIM            # 128  (spatial tokens, one per H position)
N_Vhic = R_IC             # 20   (rank tokens, one per rank mode)
N_COND = N_Uic + N_Vhic   # 148

SEQ_LEN = N_COND + N_MAIN  # 813

# flat dimensions
FLAT_U1   = N_U1  * T_DIM   # 1000
FLAT_U3   = N_U3  * H_DIM   # 2560
FLAT_G    = N_G   * R_W     # 12800
FLAT_MAIN = FLAT_U1 + FLAT_U3 + FLAT_G   # 16360

FLAT_Uic  = H_DIM * R_IC   # 128 * 20 = 2560  (elements in U_ic,  independent of tokenization)
FLAT_Vhic = R_IC  * H_DIM  # 20 * 128 = 2560  (elements in Vh_ic, independent of tokenization)
FLAT_COND = FLAT_Uic + FLAT_Vhic         # 5120


# ---------------------------------------------------------------------------
# Scalar embedder (same architecture as TimestepEmbedder, continuous input)
# ---------------------------------------------------------------------------

class ScalarEmbedder(nn.Module):
    """Embeds a continuous scalar (e.g. normalised log-nu) into hidden_size."""

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
        """Sinusoidal embedding for a continuous scalar x (B,)."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(x.device)
        args = x[:, None].float() * freqs[None]          # (B, half)
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B,)  →  (B, hidden_size)"""
        return self.mlp(self.scalar_embedding(x, self.frequency_embedding_size))


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class BurgersDiT2D(nn.Module):
    """
    Parameters
    ----------
    hidden_size : int
    depth       : int
    num_heads   : int
    mlp_ratio   : float
    """

    def __init__(
        self,
        hidden_size: int = 512,
        depth: int       = 8,
        num_heads: int   = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        H = hidden_size

        # ── Input projections ────────────────────────────────────────────────
        # main
        self.proj_U1   = nn.Linear(T_DIM, H)    # 5 tokens, dim=200 → H
        self.proj_U3   = nn.Linear(H_DIM, H)    # 20 tokens, dim=128 → H
        self.proj_G    = nn.Linear(R_W,   H)    # 640 tokens, dim=20 → H
        # condition
        self.proj_Uic  = nn.Linear(R_IC,  H)   # 128 tokens, dim=20 → H
        self.proj_Vhic = nn.Linear(H_DIM, H)   # 20 tokens, dim=128 → H

        # ── Positional embeddings (learnable) ────────────────────────────────
        self.pe_U1_rank  = nn.Embedding(R_T,   H)   # rank 0..4
        self.pe_U3_rank  = nn.Embedding(R_W,   H)   # rank 0..19
        self.pe_G_trank  = nn.Embedding(R_T,   H)   # temporal-rank dim of G
        self.pe_G_spa    = nn.Embedding(H_DIM, H)   # spatial 0..127
        self.pe_Uic_spa  = nn.Embedding(H_DIM, H)   # spatial 0..127
        self.pe_Vhic_rank= nn.Embedding(R_IC,  H)   # rank 0..19

        # ── Token type embeddings (5 types) ──────────────────────────────────
        # 0=U1, 1=U3, 2=G, 3=Uic, 4=Vhic
        self.type_embed = nn.Embedding(5, H)

        # ── Conditioning embedders ────────────────────────────────────────────
        self.t_embedder  = TimestepEmbedder(H)
        self.nu_embedder = ScalarEmbedder(H)
        self.cd_embedder = ScalarEmbedder(H)

        # ── Transformer backbone ──────────────────────────────────────────────
        self.blocks     = nn.ModuleList([DiTBlock(H, num_heads, mlp_ratio)
                                         for _ in range(depth)])
        self.final_norm = FinalLayerNorm(H)

        # ── Output heads (main tokens only, zero-init) ────────────────────────
        self.out_U1 = nn.Linear(H, T_DIM)   # → 200
        self.out_U3 = nn.Linear(H, H_DIM)   # → 128
        self.out_G  = nn.Linear(H, R_W)     # → 20

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
        for head in (self.out_U1, self.out_U3, self.out_G):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        U1:    torch.Tensor,   # (B, 200, r_T)
        U3:    torch.Tensor,   # (B, 128, r_W)
        G:     torch.Tensor,   # (B, r_T, 128, r_W)
        U_ic:  torch.Tensor,   # (B, 128, r_ic)
        Vh_ic: torch.Tensor,   # (B, r_ic, 128)
        t:     torch.Tensor,   # (B,)  diffusion timestep
        nu:    torch.Tensor,   # (B,)  normalised log-nu
        cd:    torch.Tensor,   # (B,)  normalised convection_delta
    ):
        device = U1.device
        B      = U1.shape[0]

        # ── Position index tensors ────────────────────────────────────────────
        rank_rT  = torch.arange(R_T,   device=device)    # [0..4]
        rank_rW  = torch.arange(R_W,   device=device)    # [0..19]
        rank_rIC = torch.arange(R_IC,  device=device)    # [0..19]
        spa_128  = torch.arange(H_DIM, device=device)    # [0..127]

        # G token indices: temporal-rank repeats over spatial, spatial repeats over ranks
        g_trank = rank_rT.repeat_interleave(H_DIM)   # (640,)  [0,0,..,1,1,..,...]
        g_spa   = spa_128.repeat(R_T)                 # (640,)  [0..127, 0..127, ...]

        # ── Condition tokens ─────────────────────────────────────────────────
        # U_ic (B, 128, 20) → 128 spatial tokens, dim=20 (no transpose)
        h_Uic  = (self.proj_Uic(U_ic)
                  + self.pe_Uic_spa(spa_128)
                  + self.type_embed.weight[3])              # (B, 128, H)

        # Vh_ic (B, 20, 128) → 20 rank tokens, dim=128 (no transpose)
        h_Vhic = (self.proj_Vhic(Vh_ic)
                  + self.pe_Vhic_rank(rank_rIC)
                  + self.type_embed.weight[4])              # (B, 20, H)

        # ── Main tokens ───────────────────────────────────────────────────────
        # U_1 (B, 200, r_T) → 5 rank tokens, dim=200: transpose to (B, r_T, 200)
        U1_t   = U1.transpose(1, 2)                        # (B, 5, 200)
        h_U1   = (self.proj_U1(U1_t)
                  + self.pe_U1_rank(rank_rT)
                  + self.type_embed.weight[0])              # (B, 5, H)

        # U_3 (B, 128, r_W) → 20 rank tokens, dim=128: transpose to (B, r_W, 128)
        U3_t   = U3.transpose(1, 2)                        # (B, 20, 128)
        h_U3   = (self.proj_U3(U3_t)
                  + self.pe_U3_rank(rank_rW)
                  + self.type_embed.weight[1])              # (B, 20, H)

        # G (B, r_T, 128, r_W) → (B, 640, 20)
        G_flat = G.reshape(B, R_T * H_DIM, R_W)            # (B, 640, 20)
        h_G    = (self.proj_G(G_flat)
                  + self.pe_G_trank(g_trank)
                  + self.pe_G_spa(g_spa)
                  + self.type_embed.weight[2])              # (B, 640, H)

        # ── Concatenate [COND | MAIN] ─────────────────────────────────────────
        seq = torch.cat([h_Uic, h_Vhic, h_U1, h_U3, h_G], dim=1)  # (B, 813, H)

        # ── Conditioning signal: timestep + nu + cd ──────────────────────────
        c = self.t_embedder(t) + self.nu_embedder(nu) + self.cd_embedder(cd)  # (B, H)

        # ── Transformer ───────────────────────────────────────────────────────
        for block in self.blocks:
            seq = block(seq, c)
        seq = self.final_norm(seq, c)                        # (B, 813, H)

        # ── Extract and decode main tokens ────────────────────────────────────
        s_main = seq[:, N_COND:]                             # (B, 665, H)
        s_U1   = s_main[:,             :N_U1]               # (B,   5, H)
        s_U3   = s_main[:,       N_U1 : N_U1 + N_U3]       # (B,  20, H)
        s_G    = s_main[:, N_U1 + N_U3:]                   # (B, 640, H)

        pred_U1 = self.out_U1(s_U1)                         # (B,   5, 200)
        pred_U3 = self.out_U3(s_U3)                         # (B,  20, 128)
        pred_G  = self.out_G(s_G)                           # (B, 640,  20)

        pred_U1 = pred_U1.transpose(1, 2)                   # (B, 200,   5)
        pred_U3 = pred_U3.transpose(1, 2)                   # (B, 128,  20)
        pred_G  = pred_G.reshape(B, R_T, H_DIM, R_W)       # (B,   5, 128, 20)

        return pred_U1, pred_U3, pred_G


# ---------------------------------------------------------------------------
# Wrapper: flat vector ↔ model  (compatible with src.diffusion)
# ---------------------------------------------------------------------------

class Burgers2DWrapper(nn.Module):
    """
    x_flat     layout : [U1.flat (1000) | U3.flat (2560) | G.flat (12800)]  dim=16360
    cond_flat  layout : [Uic.flat (2560) | Vhic.flat (2560)]                dim=5120

    forward(x_flat, t, cond_flat=None, nu=None)
    """

    _split_main  = [FLAT_U1, FLAT_U3, FLAT_G]
    _shapes_main = [(T_DIM, R_T), (H_DIM, R_W), (R_T, H_DIM, R_W)]

    _split_cond  = [FLAT_Uic, FLAT_Vhic]
    _shapes_cond = [(H_DIM, R_IC), (R_IC, H_DIM)]

    def __init__(self, core: BurgersDiT2D):
        super().__init__()
        self.core = core

    def forward(
        self,
        x_flat:    torch.Tensor,          # (B, 16360)
        t:         torch.Tensor,          # (B,)
        cond_flat: torch.Tensor = None,   # (B, 5120)
        nu:        torch.Tensor = None,   # (B,)  normalised log-nu
        cd:        torch.Tensor = None,   # (B,)  normalised convection_delta
    ):
        B = x_flat.shape[0]

        # unpack main
        c0, c1, c2 = x_flat.split(self._split_main, dim=1)
        U1 = c0.reshape(B, *self._shapes_main[0])    # (B, 200,  5)
        U3 = c1.reshape(B, *self._shapes_main[1])    # (B, 128, 20)
        G  = c2.reshape(B, *self._shapes_main[2])    # (B,   5, 128, 20)

        # unpack condition
        if cond_flat is None:
            cond_flat = torch.zeros(B, FLAT_COND, device=x_flat.device,
                                    dtype=x_flat.dtype)
        d0, d1 = cond_flat.split(self._split_cond, dim=1)
        U_ic  = d0.reshape(B, *self._shapes_cond[0])  # (B, 128, 20)
        Vh_ic = d1.reshape(B, *self._shapes_cond[1])  # (B,  20, 128)

        if nu is None:
            nu = torch.zeros(B, device=x_flat.device, dtype=x_flat.dtype)
        if cd is None:
            cd = torch.zeros(B, device=x_flat.device, dtype=x_flat.dtype)

        pred_U1, pred_U3, pred_G = self.core(U1, U3, G, U_ic, Vh_ic, t, nu, cd)

        return torch.cat([
            pred_U1.flatten(1),   # (B, 1000)
            pred_U3.flatten(1),   # (B, 2560)
            pred_G.flatten(1),    # (B, 12800)
        ], dim=1)                 # (B, 16360)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_burgers_2d_dit(cfg: dict) -> Burgers2DWrapper:
    core = BurgersDiT2D(
        hidden_size = cfg.get('hidden_size', 512),
        depth       = cfg.get('depth',       8),
        num_heads   = cfg.get('num_heads',   8),
        mlp_ratio   = cfg.get('mlp_ratio',   4.0),
    )
    return Burgers2DWrapper(core)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    B = 2
    wrapper = build_burgers_2d_dit({})
    n_params = sum(p.numel() for p in wrapper.parameters())
    print(f'Model params: {n_params:,}')
    print(f'FLAT_MAIN={FLAT_MAIN}  FLAT_COND={FLAT_COND}  SEQ_LEN={SEQ_LEN}')

    x     = torch.randn(B, FLAT_MAIN)
    cond  = torch.randn(B, FLAT_COND)
    t     = torch.randint(0, 1000, (B,))
    nu    = torch.randn(B)
    cd    = torch.randn(B)
    out   = wrapper(x, t, cond_flat=cond, nu=nu, cd=cd)
    print(f'Input  x:    {tuple(x.shape)}')
    print(f'Output out:  {tuple(out.shape)}  (should match x)')
    assert out.shape == x.shape, 'Shape mismatch!'
    print('Forward pass OK.')
