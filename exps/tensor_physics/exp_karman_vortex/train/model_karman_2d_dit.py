"""
model_karman_2d_dit.py — Conditional DiT for 2D Kármán-vortex Tucker factor diffusion.

Tucker rank = [r_T=10, r_X=128, r_Y=30] on (T=200, X=128, Y=128).
Condition: initial frame (U_ic, Vh_ic) + five scalars (niu, cx, cy, r, Re).

Factor shapes (after absorbing U_X into G):
  U_T : (200, r_T)
  U_Y : (128, r_Y)
  G   : (r_T, 128, r_Y)        G = C ×_2 U_X (r_X = 128 full rank)

Token layout (1478 tokens total):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  CONDITION (158, not noised)                                        │
  │    U_ic  (128, 30) → 128 spatial tokens, dim=30     type=3         │
  │    Vh_ic (30, 128) →  30 rank tokens,    dim=128    type=4         │
  ├─────────────────────────────────────────────────────────────────────┤
  │  MAIN (1320, noised / denoised)                                     │
  │    U_T (200,10)ᵀ → 10 rank tokens,    dim=200       type=0         │
  │    U_Y (128,30)ᵀ → 30 rank tokens,    dim=128       type=1         │
  │    G   (10,128,30) → 1280 spatial tokens, dim=30    type=2         │
  └─────────────────────────────────────────────────────────────────────┘

Scalar conditioning:
  c = t_embedder(t) + niu_emb + cx_emb + cy_emb + r_emb + re_emb

x_flat      : [U_T.flat (2000) | U_Y.flat (3840) | G.flat (38400)] dim=44240
cond_flat   : [U_ic.flat (3840) | Vh_ic.flat (3840)]              dim=7680
"""

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, '/home/x-jlyu5/jinhua/factor_diffusion/video')
from models.cp_dit import (
    Mlp, modulate,
    TimestepEmbedder, SDPAAttention, DiTBlock, FinalLayerNorm,
)

# ---------------------------------------------------------------------------
# Dimension constants (rank=[10,128,30], r_ic=30, T=200, X=Y=128)
# ---------------------------------------------------------------------------
R_T, R_X, R_Y = 10, 128, 30
R_IC  = 30
T_DIM = 200
H_DIM = 128   # shared spatial dim (X = Y = H_DIM)

# main tokens
N_UT   = R_T               # 10
N_UY   = R_Y               # 30
N_G    = R_T * H_DIM       # 1280
N_MAIN = N_UT + N_UY + N_G # 1320

# condition tokens
N_Uic  = H_DIM             # 128
N_Vhic = R_IC              # 30
N_COND = N_Uic + N_Vhic    # 158

SEQ_LEN = N_COND + N_MAIN  # 1478

# flat sizes
FLAT_UT   = N_UT * T_DIM           # 2000
FLAT_UY   = N_UY * H_DIM           # 3840
FLAT_G    = N_G  * R_Y             # 38400
FLAT_MAIN = FLAT_UT + FLAT_UY + FLAT_G   # 44240

FLAT_Uic  = H_DIM * R_IC           # 3840
FLAT_Vhic = R_IC  * H_DIM          # 3840
FLAT_COND = FLAT_Uic + FLAT_Vhic   # 7680


# ---------------------------------------------------------------------------
# Scalar embedder (identical to burgers)
# ---------------------------------------------------------------------------

class ScalarEmbedder(nn.Module):
    """Embeds a continuous scalar into hidden_size via sinusoidal + MLP."""

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
        half  = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(x.device)
        args = x[:, None].float() * freqs[None]
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.scalar_embedding(x, self.frequency_embedding_size))


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class KarmanDiT2D(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        depth: int       = 8,
        num_heads: int   = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        H = hidden_size

        # Input projections
        self.proj_UT   = nn.Linear(T_DIM, H)   # 10 tokens, dim=200
        self.proj_UY   = nn.Linear(H_DIM, H)   # 30 tokens, dim=128
        self.proj_G    = nn.Linear(R_Y,   H)   # 1280 tokens, dim=30
        self.proj_Uic  = nn.Linear(R_IC,  H)   # 128 tokens, dim=30
        self.proj_Vhic = nn.Linear(H_DIM, H)   # 30 tokens, dim=128

        # Positional embeddings
        self.pe_UT_rank   = nn.Embedding(R_T,   H)
        self.pe_UY_rank   = nn.Embedding(R_Y,   H)
        self.pe_G_trank   = nn.Embedding(R_T,   H)
        self.pe_G_spa     = nn.Embedding(H_DIM, H)
        self.pe_Uic_spa   = nn.Embedding(H_DIM, H)
        self.pe_Vhic_rank = nn.Embedding(R_IC,  H)

        # Token types: 0=UT, 1=UY, 2=G, 3=Uic, 4=Vhic
        self.type_embed = nn.Embedding(5, H)

        # Conditioning
        self.t_embedder   = TimestepEmbedder(H)
        self.niu_embedder = ScalarEmbedder(H)
        self.cx_embedder  = ScalarEmbedder(H)
        self.cy_embedder  = ScalarEmbedder(H)
        self.r_embedder   = ScalarEmbedder(H)
        self.re_embedder  = ScalarEmbedder(H)

        # Transformer
        self.blocks     = nn.ModuleList([DiTBlock(H, num_heads, mlp_ratio)
                                          for _ in range(depth)])
        self.final_norm = FinalLayerNorm(H)

        # Output heads
        self.out_UT = nn.Linear(H, T_DIM)
        self.out_UY = nn.Linear(H, H_DIM)
        self.out_G  = nn.Linear(H, R_Y)

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
        for head in (self.out_UT, self.out_UY, self.out_G):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        U_T:   torch.Tensor,  # (B, 200, r_T)
        U_Y:   torch.Tensor,  # (B, 128, r_Y)
        G:     torch.Tensor,  # (B, r_T, 128, r_Y)
        U_ic:  torch.Tensor,  # (B, 128, r_ic)
        Vh_ic: torch.Tensor,  # (B, r_ic, 128)
        t:     torch.Tensor,  # (B,)
        niu:   torch.Tensor,  # (B,)
        cx:    torch.Tensor,  # (B,)
        cy:    torch.Tensor,  # (B,)
        r:     torch.Tensor,  # (B,)
        re:    torch.Tensor,  # (B,)  normalised log-Re
    ):
        device = U_T.device
        B = U_T.shape[0]

        rank_rT  = torch.arange(R_T,  device=device)
        rank_rY  = torch.arange(R_Y,  device=device)
        rank_rIC = torch.arange(R_IC, device=device)
        spa_128  = torch.arange(H_DIM, device=device)

        # G token positions
        g_trank = rank_rT.repeat_interleave(H_DIM)   # (1280,)
        g_spa   = spa_128.repeat(R_T)                 # (1280,)

        # Condition tokens
        h_Uic  = (self.proj_Uic(U_ic)
                  + self.pe_Uic_spa(spa_128)
                  + self.type_embed.weight[3])
        h_Vhic = (self.proj_Vhic(Vh_ic)
                  + self.pe_Vhic_rank(rank_rIC)
                  + self.type_embed.weight[4])

        # Main tokens
        UT_t = U_T.transpose(1, 2)                    # (B, r_T, 200)
        h_UT = (self.proj_UT(UT_t)
                + self.pe_UT_rank(rank_rT)
                + self.type_embed.weight[0])

        UY_t = U_Y.transpose(1, 2)                    # (B, r_Y, 128)
        h_UY = (self.proj_UY(UY_t)
                + self.pe_UY_rank(rank_rY)
                + self.type_embed.weight[1])

        G_flat = G.reshape(B, R_T * H_DIM, R_Y)       # (B, 1280, r_Y)
        h_G    = (self.proj_G(G_flat)
                  + self.pe_G_trank(g_trank)
                  + self.pe_G_spa(g_spa)
                  + self.type_embed.weight[2])

        seq = torch.cat([h_Uic, h_Vhic, h_UT, h_UY, h_G], dim=1)

        c = (self.t_embedder(t)
             + self.niu_embedder(niu)
             + self.cx_embedder(cx)
             + self.cy_embedder(cy)
             + self.r_embedder(r)
             + self.re_embedder(re))

        for block in self.blocks:
            seq = block(seq, c)
        seq = self.final_norm(seq, c)

        s_main = seq[:, N_COND:]
        s_UT   = s_main[:,                :N_UT]
        s_UY   = s_main[:,      N_UT : N_UT + N_UY]
        s_G    = s_main[:, N_UT + N_UY:]

        pred_UT = self.out_UT(s_UT).transpose(1, 2)            # (B, 200, r_T)
        pred_UY = self.out_UY(s_UY).transpose(1, 2)            # (B, 128, r_Y)
        pred_G  = self.out_G(s_G).reshape(B, R_T, H_DIM, R_Y)  # (B, r_T, 128, r_Y)

        return pred_UT, pred_UY, pred_G


# ---------------------------------------------------------------------------
# Flat wrapper
# ---------------------------------------------------------------------------

class Karman2DWrapper(nn.Module):
    """
    x_flat    : [UT.flat | UY.flat | G.flat]          dim=44240
    cond_flat : [U_ic.flat | Vh_ic.flat]              dim=7680
    """

    _split_main  = [FLAT_UT, FLAT_UY, FLAT_G]
    _shapes_main = [(T_DIM, R_T), (H_DIM, R_Y), (R_T, H_DIM, R_Y)]
    _split_cond  = [FLAT_Uic, FLAT_Vhic]
    _shapes_cond = [(H_DIM, R_IC), (R_IC, H_DIM)]

    def __init__(self, core: KarmanDiT2D):
        super().__init__()
        self.core = core

    def forward(
        self,
        x_flat:    torch.Tensor,
        t:         torch.Tensor,
        cond_flat: torch.Tensor = None,
        niu:       torch.Tensor = None,
        cx:        torch.Tensor = None,
        cy:        torch.Tensor = None,
        r:         torch.Tensor = None,
        re:        torch.Tensor = None,
    ):
        B = x_flat.shape[0]

        c0, c1, c2 = x_flat.split(self._split_main, dim=1)
        U_T = c0.reshape(B, *self._shapes_main[0])
        U_Y = c1.reshape(B, *self._shapes_main[1])
        G   = c2.reshape(B, *self._shapes_main[2])

        if cond_flat is None:
            cond_flat = torch.zeros(B, FLAT_COND, device=x_flat.device,
                                    dtype=x_flat.dtype)
        d0, d1 = cond_flat.split(self._split_cond, dim=1)
        U_ic  = d0.reshape(B, *self._shapes_cond[0])
        Vh_ic = d1.reshape(B, *self._shapes_cond[1])

        zeros = torch.zeros(B, device=x_flat.device, dtype=x_flat.dtype)
        if niu is None: niu = zeros
        if cx  is None: cx  = zeros
        if cy  is None: cy  = zeros
        if r   is None: r   = zeros
        if re  is None: re  = zeros

        pred_UT, pred_UY, pred_G = self.core(U_T, U_Y, G, U_ic, Vh_ic,
                                              t, niu, cx, cy, r, re)

        return torch.cat([
            pred_UT.flatten(1),
            pred_UY.flatten(1),
            pred_G.flatten(1),
        ], dim=1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_karman_2d_dit(cfg: dict) -> Karman2DWrapper:
    core = KarmanDiT2D(
        hidden_size = cfg.get('hidden_size', 512),
        depth       = cfg.get('depth',       8),
        num_heads   = cfg.get('num_heads',   8),
        mlp_ratio   = cfg.get('mlp_ratio',   4.0),
    )
    return Karman2DWrapper(core)


if __name__ == '__main__':
    B = 2
    wrapper = build_karman_2d_dit({})
    n_params = sum(p.numel() for p in wrapper.parameters())
    print(f'Model params: {n_params:,}')
    print(f'FLAT_MAIN={FLAT_MAIN}  FLAT_COND={FLAT_COND}  SEQ_LEN={SEQ_LEN}')

    x    = torch.randn(B, FLAT_MAIN)
    cond = torch.randn(B, FLAT_COND)
    t    = torch.randint(0, 1000, (B,))
    niu  = torch.randn(B)
    cx   = torch.randn(B)
    cy   = torch.randn(B)
    r    = torch.randn(B)
    re   = torch.randn(B)
    out  = wrapper(x, t, cond_flat=cond, niu=niu, cx=cx, cy=cy, r=r, re=re)
    print(f'Input  x: {tuple(x.shape)}')
    print(f'Output:   {tuple(out.shape)}')
    assert out.shape == x.shape, 'Shape mismatch!'
    print('Forward pass OK.')
