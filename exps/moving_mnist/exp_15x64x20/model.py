"""
Tucker3PartsDiT -- Diffusion model for Tucker 3-part factors (no patchification).

The 3 parts are G stored directly in shard (U_2 already absorbed into C):
  G shape: (r_T, H=64, r_W)

Tokenization:
  U_1 : (B, T=20,  r_T)    -> r_T    tokens, proj Linear(T,  H)
  U_3 : (B, W=64,  r_W)    -> r_W    tokens, proj Linear(W,  H)
  G   : (B, r_TxH, r_W)    -> r_T*H  tokens, proj Linear(r_W, H)
        Positional embedding: pe_G_i(i) + pe_G_j(j)  where token k = i*H + j

seq_len = r_T + r_W + r_T*H

Flat vector layout (for wrapper):
  [U_1.flat | U_3.flat | G.flat]
  sizes: [T*r_T, W*r_W, r_T*H*r_W]

Reconstruction:
  G = G_flat.reshape(r_T, H, r_W)
  video[t,h,w] = Sigma_{a,c} G[a,h,c] * U_1[t,a] * U_3[w,c]
  i.e. einsum('bahc,bta,bwc->bthw', G, U_1, U_3) -> (B, T, H, W)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from models.cp_dit import (
    modulate,
    TimestepEmbedder, SDPAAttention, DiTBlock, FinalLayerNorm,
)


class Tucker3PartsDiT(nn.Module):
    """
    Parameters
    ----------
    T    : temporal frames  (default 20)
    H    : height           (default 64)
    W    : width            (default 64)
    r_T  : Tucker rank dim-0  (U_1 columns, G dim-0)
    r_W  : Tucker rank dim-2  (U_3 columns, G dim-2)
    hidden_size, depth, num_heads, mlp_ratio : transformer hyperparams
    """

    def __init__(
        self,
        T: int   = 20,
        H: int   = 64,
        W: int   = 64,
        r_T: int = 15,
        r_W: int = 20,
        hidden_size: int = 512,
        depth: int       = 8,
        num_heads: int   = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.T, self.H, self.W = T, H, W
        self.r_T, self.r_W = r_T, r_W
        self.seq_len = r_T + r_W + r_T * H

        HS = hidden_size

        # -- Input projections --------------------------------------------------
        self.proj_U1 = nn.Linear(T,   HS)   # U_1 col: dim=T
        self.proj_U3 = nn.Linear(W,   HS)   # U_3 col: dim=W
        self.proj_G  = nn.Linear(r_W, HS)   # G row:   dim=r_W

        # -- Positional embeddings ----------------------------------------------
        self.pe_U1  = nn.Embedding(r_T, HS)   # r_T positions
        self.pe_U3  = nn.Embedding(r_W, HS)   # r_W positions
        self.pe_G_i = nn.Embedding(r_T, HS)   # G dim-0 (temporal) position
        self.pe_G_j = nn.Embedding(H,   HS)   # G dim-1 (height)   position

        # -- Token type embedding -----------------------------------------------
        self.type_embed = nn.Embedding(3, HS)   # 0=U1, 1=U3, 2=G

        # -- Timestep conditioning ----------------------------------------------
        self.t_embedder = TimestepEmbedder(HS)

        # -- Transformer backbone -----------------------------------------------
        self.blocks = nn.ModuleList([
            DiTBlock(HS, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_norm = FinalLayerNorm(HS)

        # -- Output heads (zero-init) -------------------------------------------
        self.out_U1 = nn.Linear(HS, T)    # (B, r_T, T) -> transpose -> (B, T, r_T)
        self.out_U3 = nn.Linear(HS, W)    # (B, r_W, W) -> transpose -> (B, W, r_W)
        self.out_G  = nn.Linear(HS, r_W)  # (B, r_T*H, r_W)

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
        U_1:   torch.Tensor,   # (B, T,   r_T)
        U_3:   torch.Tensor,   # (B, W,   r_W)
        G_flat: torch.Tensor,  # (B, r_T*H, r_W)
        t:     torch.Tensor,   # (B,)
    ):
        device = U_1.device
        r_T, r_W, H = self.r_T, self.r_W, self.H

        # -- Position indices ---------------------------------------------------
        idx_U1 = torch.arange(r_T, device=device)
        idx_U3 = torch.arange(r_W, device=device)
        i_idx  = torch.arange(r_T, device=device).repeat_interleave(H)
        j_idx  = torch.arange(H,   device=device).repeat(r_T)

        # -- Token embeddings ---------------------------------------------------
        h_U1 = (self.proj_U1(U_1.transpose(-1, -2))   # (B, r_T, HS)
                + self.pe_U1(idx_U1)
                + self.type_embed.weight[0])

        h_U3 = (self.proj_U3(U_3.transpose(-1, -2))   # (B, r_W, HS)
                + self.pe_U3(idx_U3)
                + self.type_embed.weight[1])

        h_G = (self.proj_G(G_flat)                    # (B, r_T*H, HS)
               + self.pe_G_i(i_idx)
               + self.pe_G_j(j_idx)
               + self.type_embed.weight[2])

        seq = torch.cat([h_U1, h_U3, h_G], dim=1)

        c = self.t_embedder(t)
        for block in self.blocks:
            seq = block(seq, c)
        seq = self.final_norm(seq, c)

        # -- Split output sequence ----------------------------------------------
        s_U1 = seq[:,       :r_T]
        s_U3 = seq[:, r_T  :r_T + r_W]
        s_G  = seq[:, r_T + r_W:]

        out_U1 = self.out_U1(s_U1).transpose(-1, -2)  # (B, T,  r_T)
        out_U3 = self.out_U3(s_U3).transpose(-1, -2)  # (B, W,  r_W)
        out_G  = self.out_G(s_G)                       # (B, r_T*H, r_W)

        return out_U1, out_U3, out_G


# ---------------------------------------------------------------------------
# Wrapper: flat vector <-> model
# ---------------------------------------------------------------------------

class Tucker3PartsWrapper(nn.Module):
    """
    x_flat layout: [U_1.flat | U_3.flat | G.flat]
    sizes: [T*r_T, W*r_W, r_T*H*r_W]
    """

    def __init__(self, core: Tucker3PartsDiT):
        super().__init__()
        self.core = core
        T, H, W = core.T, core.H, core.W
        r_T, r_W = core.r_T, core.r_W
        self._split  = [T * r_T, W * r_W, r_T * H * r_W]
        self._shapes = [(T, r_T), (W, r_W), (r_T * H, r_W)]

    def forward(self, x_flat: torch.Tensor, t: torch.Tensor, y=None):
        B = x_flat.shape[0]
        chunks = x_flat.split(self._split, dim=1)
        U_1    = chunks[0].reshape(B, *self._shapes[0])
        U_3    = chunks[1].reshape(B, *self._shapes[1])
        G_flat = chunks[2].reshape(B, *self._shapes[2])

        o1, o3, oG = self.core(U_1, U_3, G_flat, t)

        return torch.cat([
            o1.flatten(1),
            o3.flatten(1),
            oG.flatten(1),
        ], dim=1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_tucker_3parts_dit(cfg: dict) -> Tucker3PartsWrapper:
    core = Tucker3PartsDiT(
        T           = cfg.get('T',    20),
        H           = cfg.get('H',    64),
        W           = cfg.get('W',    64),
        r_T         = cfg.get('r_T',  15),
        r_W         = cfg.get('r_W',  20),
        hidden_size = cfg.get('hidden_size', 512),
        depth       = cfg.get('depth',       8),
        num_heads   = cfg.get('num_heads',   8),
        mlp_ratio   = cfg.get('mlp_ratio',   4.0),
    )
    return Tucker3PartsWrapper(core)
