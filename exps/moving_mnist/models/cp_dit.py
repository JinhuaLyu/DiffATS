"""
CPFactorDiT -- DiT-style transformer for CP factor diffusion.

Input per sample: three factor matrices from channel-wise CP decomposition:
  x_temporal : (B, 3, T, R)      # mode0: T time tokens, each R-dim
  x_spatial  : (B, 3, N_patch, R)  # mode1: 400 patch-location tokens, each R-dim
  x_content  : (B, 3, R, d_c)    # mode2.T: R rank tokens, each d_c(=48)-dim

Tokenization:
  temporal tokens  -> project R   -> hidden_size, add 1D temporal PE (learnable, 0..T-1)
  spatial  tokens  -> project R   -> hidden_size, add 2D spatial  PE (learnable, 20x20 flattened)
  content  tokens  -> project d_c -> hidden_size, add 1D rank-order PE (learnable, 0..R-1)

All tokens are tagged with a "token type" embedding (0/1/2).
Sequence is (Bx3, T+N_patch+R, hidden_size); the three channels are folded into batch dim.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """Simple MLP with GELU activation (replaces timm.Mlp)."""
    def __init__(self, in_features, hidden_features=None, act_layer=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer() if act_layer is not None else nn.GELU(approximate="tanh")
        self.fc2  = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


# ---------------------------------------------------------------------------
# Helpers copied/adapted from dit_models.py
# ---------------------------------------------------------------------------

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class SDPAAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        H, hd   = self.num_heads, self.head_dim
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        def to_heads(t):
            return t.view(B, T, H, hd).permute(0, 2, 1, 3).reshape(B * H, T, hd)
        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.reshape(B, H, T, hd).permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = SDPAAttention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp   = Mlp(in_features=hidden_size, hidden_features=mlp_hidden,
                         act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayerNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm    = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN   = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        return modulate(self.norm(x), shift, scale)


# ---------------------------------------------------------------------------
# CPFactorDiT
# ---------------------------------------------------------------------------

class CPFactorDiT(nn.Module):
    """
    DiT for CP factor diffusion.

    Parameters
    ----------
    T         : int   -- number of temporal frames (max_frames, e.g. 40)
    N_patch   : int   -- number of spatial patches (e.g. 400 = 20x20)
    R         : int   -- CP rank (e.g. 100)
    d_content : int   -- patch-content token dim (48 = 6x8 flattened)
    hidden_size : int -- transformer hidden dimension
    depth     : int   -- number of DiTBlock layers
    num_heads : int   -- number of attention heads
    mlp_ratio : float -- MLP expansion ratio
    n_channels: int   -- number of colour channels (3 for BGR)
    learn_sigma: bool -- if True, output doubled for variance prediction
    """

    def __init__(
        self,
        T: int         = 40,
        N_patch: int   = 400,
        R: int         = 100,
        d_content: int = 48,
        hidden_size: int = 512,
        depth: int       = 12,
        num_heads: int   = 8,
        mlp_ratio: float = 4.0,
        n_channels: int  = 3,
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.T          = T
        self.N_patch    = N_patch
        self.R          = R
        self.d_content  = d_content
        self.hidden_size = hidden_size
        self.n_channels = n_channels
        self.learn_sigma = learn_sigma
        self.seq_len    = T + N_patch + R     # total tokens per channel

        # -- Input projections ----------------------------------------------
        self.temporal_proj = nn.Linear(R, hidden_size)           # (T, R)  -> (T, H)
        self.spatial_proj  = nn.Linear(R, hidden_size)           # (N, R)  -> (N, H)
        self.content_proj  = nn.Linear(d_content, hidden_size)   # (R, 48) -> (R, H)

        # -- Positional embeddings ------------------------------------------
        # temporal: 1D learnable, index 0..T-1
        self.temporal_pe  = nn.Embedding(T, hidden_size)
        # spatial: 2D learnable, flattened 20x20 = 400
        self.spatial_pe   = nn.Embedding(N_patch, hidden_size)
        # content: 1D learnable by rank order, index 0..R-1
        self.content_pe   = nn.Embedding(R, hidden_size)

        # -- Token type embedding (0=temporal, 1=spatial, 2=content) -------
        self.type_embed = nn.Embedding(3, hidden_size)

        # -- Timestep conditioning ------------------------------------------
        self.t_embedder = TimestepEmbedder(hidden_size)

        # -- Transformer blocks ---------------------------------------------
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_norm = FinalLayerNorm(hidden_size)

        # -- Output projections ---------------------------------------------
        out_scale = 2 if learn_sigma else 1
        self.temporal_out = nn.Linear(hidden_size, R * out_scale)
        self.spatial_out  = nn.Linear(hidden_size, R * out_scale)
        self.content_out  = nn.Linear(hidden_size, d_content * out_scale)

        self._init_weights()

    def _init_weights(self):
        # Zero-init output projections (same as reference DiT)
        nn.init.zeros_(self.temporal_out.weight)
        nn.init.zeros_(self.temporal_out.bias)
        nn.init.zeros_(self.spatial_out.weight)
        nn.init.zeros_(self.spatial_out.bias)
        nn.init.zeros_(self.content_out.weight)
        nn.init.zeros_(self.content_out.bias)
        # Zero-init adaLN modulations
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)

    def _build_positions(self, device):
        t_idx = torch.arange(self.T,       device=device)   # (T,)
        s_idx = torch.arange(self.N_patch, device=device)   # (N,)
        c_idx = torch.arange(self.R,       device=device)   # (R,)
        return t_idx, s_idx, c_idx

    def forward(
        self,
        x_temporal: torch.Tensor,   # (B, 3, T, R)
        x_spatial:  torch.Tensor,   # (B, 3, N, R)
        x_content:  torch.Tensor,   # (B, 3, R, d_c)
        t:          torch.Tensor,   # (B,) diffusion timestep
    ):
        B = x_temporal.shape[0]
        C = self.n_channels
        device = x_temporal.device

        # -- Fold channel dim into batch: (B*C, ...) ------------------------
        xt = x_temporal.reshape(B * C, self.T,       self.R)           # (B*C, T, R)
        xs = x_spatial.reshape(B * C,  self.N_patch, self.R)           # (B*C, N, R)
        xc = x_content.reshape(B * C,  self.R,       self.d_content)   # (B*C, R, d_c)

        # -- Project to hidden_size -----------------------------------------
        ht = self.temporal_proj(xt)    # (B*C, T, H)
        hs = self.spatial_proj(xs)     # (B*C, N, H)
        hc = self.content_proj(xc)     # (B*C, R, H)

        # -- Add positional embeddings --------------------------------------
        t_idx, s_idx, c_idx = self._build_positions(device)
        ht = ht + self.temporal_pe(t_idx).unsqueeze(0)   # (1, T, H)
        hs = hs + self.spatial_pe(s_idx).unsqueeze(0)    # (1, N, H)
        hc = hc + self.content_pe(c_idx).unsqueeze(0)    # (1, R, H)

        # -- Add token type embeddings --------------------------------------
        type_t = self.type_embed(torch.zeros(1, dtype=torch.long, device=device))   # (1, H)
        type_s = self.type_embed(torch.ones(1,  dtype=torch.long, device=device))   # (1, H)
        type_c = self.type_embed(torch.full((1,), 2, dtype=torch.long, device=device))  # (1, H)
        ht = ht + type_t
        hs = hs + type_s
        hc = hc + type_c

        # -- Concatenate into one sequence ----------------------------------
        seq = torch.cat([ht, hs, hc], dim=1)   # (B*C, T+N+R, H)

        # -- Timestep conditioning (broadcast over channels) ----------------
        t_rep = t.repeat_interleave(C)           # (B*C,)
        c_emb = self.t_embedder(t_rep)            # (B*C, H)

        # -- Transformer blocks ---------------------------------------------
        for block in self.blocks:
            seq = block(seq, c_emb)

        seq = self.final_norm(seq, c_emb)        # (B*C, T+N+R, H)

        # -- Split back into three parts ------------------------------------
        T, N, R = self.T, self.N_patch, self.R
        seq_t = seq[:, :T]                        # (B*C, T, H)
        seq_s = seq[:, T: T + N]                  # (B*C, N, H)
        seq_c = seq[:, T + N:]                    # (B*C, R, H)

        # -- Output projections ---------------------------------------------
        out_t = self.temporal_out(seq_t)           # (B*C, T, R[*2])
        out_s = self.spatial_out(seq_s)            # (B*C, N, R[*2])
        out_c = self.content_out(seq_c)            # (B*C, R, d_c[*2])

        # -- Unfold channel dim ---------------------------------------------
        if self.learn_sigma:
            out_t = out_t.reshape(B, C, T, R, 2)
            out_s = out_s.reshape(B, C, N, R, 2)
            out_c = out_c.reshape(B, C, R, self.d_content, 2)
        else:
            out_t = out_t.reshape(B, C, T, R)
            out_s = out_s.reshape(B, C, N, R)
            out_c = out_c.reshape(B, C, R, self.d_content)

        return out_t, out_s, out_c
