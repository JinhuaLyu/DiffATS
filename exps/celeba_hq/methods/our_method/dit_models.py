# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


class ChannelWisePatchEmbed(nn.Module):
    """
    First embed each channel separately, then apply a learnable per-channel scale and take a weighted sum.
    Each channel uses its own Linear(ph*pw -> embed_dim), avoiding direct channel mixing inside PatchEmbed.
    The interface is compatible with timm PatchEmbed (grid_size, num_patches, patch_size attributes).
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim, bias=True):
        super().__init__()
        H, W   = img_size   if isinstance(img_size,   (tuple, list)) else (img_size,   img_size)
        ph, pw = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        self.patch_size  = (ph, pw)
        self.grid_size   = (H // ph, W // pw)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.in_chans    = in_chans
        # Independent linear mapping for each channel
        self.projs = nn.ModuleList([
            nn.Linear(ph * pw, embed_dim, bias=bias) for _ in range(in_chans)
        ])
        # Learnable per-channel scale, initialized to 1 (so the weighted sum reduces to a simple sum initially)
        self.channel_scale = nn.Parameter(torch.ones(in_chans, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw     = self.patch_size
        Gh, Gw     = self.grid_size
        # Extract patches: (B, C, Gh, ph, Gw, pw) -> (B, C, T, ph*pw)
        x = x.reshape(B, C, Gh, ph, Gw, pw)
        x = x.permute(0, 1, 2, 4, 3, 5).reshape(B, C, Gh * Gw, ph * pw)
        # Embed each channel independently -> (B, C, T, D)
        out = torch.stack([self.projs[c](x[:, c]) for c in range(C)], dim=1)
        # Apply per-channel scaling, then sum along the channel dimension -> (B, T, D)
        out = out * self.channel_scale.unsqueeze(0).unsqueeze(2)  # (1,C,1,D)
        return out.sum(dim=1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def tuple_to_grid(grid_size):
    """
    Convert (Gh, Gw) to a grid of coordinates (2, Gh*Gw)
    grid[0]: x coordinates (column direction), length Gh*Gw
    grid[1]: y coordinates (row direction), length Gh*Gw
    """
    Gh, Gw = grid_size
    grid_h = np.arange(Gh, dtype=np.float32)
    grid_w = np.arange(Gw, dtype=np.float32)
    mesh = np.meshgrid(grid_w, grid_h)         # (2, Gh, Gw)
    grid = np.stack(mesh, axis=0).reshape(2, -1)  # (2, Gh*Gw)
    return grid

def build_col_neighbor_mask(grid_size, device, dtype=torch.float32, radius=1, wrap=False):
    """
    Only allow attention to the same column and adjacent columns (|Deltacol| <= radius).
    grid_size: (Gh, Gw)
    Returns: additive mask of shape (1, 1, T, T); allowed=0, disallowed=-inf (can be added to attention logits)
    """
    Gh, Gw = grid_size
    T = Gh * Gw
    idx = torch.arange(T, device=device)
    rows = idx // Gw
    cols = idx % Gw

    # Compute column difference
    col_i = cols[:, None]                # (T,1)
    col_j = cols[None, :]                # (1,T)
    if wrap:
        # Wrap-around adjacency (e.g., leftmost column's neighbors include rightmost column)
        diff = torch.minimum((col_i - col_j).abs(), Gw - (col_i - col_j).abs())
    else:
        diff = (col_i - col_j).abs()

    allowed = diff <= radius             # (T,T) bool
    mask = torch.zeros((T, T), device=device, dtype=dtype)
    neg_inf = torch.finfo(dtype).min
    mask = mask.masked_fill(~allowed, neg_inf)
    # Shape adaptation (B*nH, T, T) or (1, T, T): use (1,1,T,T) for maximum generality (will broadcast to all heads/batches)
    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
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
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class SDPAAttention(nn.Module):
    """
    Minimal replacement version: implements multi-head self-attention (MSA) using PyTorch 2.x's scaled_dot_product_attention.
    Shape conventions:
      Input x: [B, T, D]
      Output y: [B, T, D]
    Supports additive/boolean attn_mask (broadcast to [B*H, Tq, Tk])
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        H = self.num_heads
        qkv = self.qkv(x)                           # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, T, D] -> [B, H, T, d]
        def reshape_heads(t):
            return t.view(B, T, H, self.head_dim).permute(0, 2, 1, 3).contiguous()

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # PyTorch SDPA expects [B*H, T, d]
        q = q.reshape(B * H, T, self.head_dim)
        k = k.reshape(B * H, T, self.head_dim)
        v = v.reshape(B * H, T, self.head_dim)

        # Handle mask: supports (1,1,T,T) or (T,T). Will broadcast to [B*H, T, T]
        am = None
        if attn_mask is not None:
            # If it's an additive mask (allowed=0, disallowed=-inf), pass directly (some versions fall back to non-Flash kernels)
            if attn_mask.dtype.is_floating_point:
                # Make shape [1, T, T] for broadcasting
                am = attn_mask
                if am.dim() == 4:   # [1,1,T,T] -> [1,T,T]
                    am = am.squeeze(0).squeeze(0)
                elif am.dim() == 2: # [T,T]
                    pass
                else:
                    raise ValueError(f"Unsupported attn_mask shape: {attn_mask.shape}")
            else:
                # Boolean mask: True=keep or True=mask? SDPA uses "True=keep"; mixing additive/boolean has ambiguity;
                # Safest approach: convert bool to additive: allowed=True -> 0; disallowed=False -> -inf
                am_bool = attn_mask
                if am_bool.dim() == 4:
                    am_bool = am_bool.squeeze(0).squeeze(0)  # [T,T]
                neg_inf = torch.finfo(q.dtype).min
                am = torch.where(am_bool, torch.zeros_like(am_bool, dtype=q.dtype), torch.full_like(am_bool, neg_inf, dtype=q.dtype))

        # SDPA (scale is built-in, no need for manual /sqrt(d))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=0.0, is_causal=False)
        # [B*H, T, d] -> [B, T, D]
        y = y.reshape(B, H, T, self.head_dim).permute(0, 2, 1, 3).contiguous().view(B, T, D)

        y = self.proj(y)
        y = self.proj_drop(y)
        return y
#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Optionally adds a cross-attention sub-layer that attends to fixed dictionary tokens.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_dict_cond=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SDPAAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Optional cross-attention to fixed dictionary tokens
        self.use_dict_cond = use_dict_cond
        if use_dict_cond:
            self.cross_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_q    = nn.Linear(hidden_size, hidden_size, bias=True)
            self.cross_kv   = nn.Linear(hidden_size, hidden_size * 2, bias=True)
            self.cross_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            # Tanh gate -- zero-init keeps no cross-attention at start; grows during training
            self.cross_gate = nn.Parameter(torch.zeros(hidden_size))

    def _cross_attn(self, x, dict_tokens):
        """x: (B,T,D)  dict_tokens: (1,M,D) -- attend x to dict_tokens as K,V."""
        B, T, D_h = x.shape
        M = dict_tokens.shape[1]
        H = self.attn.num_heads
        hd = D_h // H

        q  = self.cross_q(self.cross_norm(x))              # (B, T, D)
        kv = self.cross_kv(dict_tokens.expand(B, -1, -1))  # (B, M, 2D)
        k, v = kv.chunk(2, dim=-1)                          # each (B, M, D)

        def to_heads(t, S):
            return t.view(B, S, H, hd).permute(0, 2, 1, 3).reshape(B * H, S, hd)

        q = to_heads(q, T)
        k = to_heads(k, M)
        v = to_heads(v, M)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)   # (B*H, T, hd)
        out = out.reshape(B, H, T, hd).permute(0, 2, 1, 3).contiguous().view(B, T, D_h)
        out = self.cross_proj(out)                                        # (B, T, D)
        return out

    def forward(self, x, c, attn_mask=None, dict_tokens=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        # Self-attention
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            attn_mask=attn_mask
        )
        # Cross-attention to dictionary tokens (gated residual)
        if self.use_dict_cond and dict_tokens is not None:
            x = x + torch.tanh(self.cross_gate) * self._cross_attn(x, dict_tokens)
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    patch_size = (ph, pw)
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # test patch_size is int or tuple/list of length 2
        if isinstance(patch_size, int):
            self.patch_h, self.patch_w = patch_size, patch_size
        elif isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            self.patch_h, self.patch_w = patch_size
        else:
            raise ValueError("patch_size must be int or tuple/list of length 2")

        out_dim = self.patch_h * self.patch_w * out_channels
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x) # (N, patch_h * patch_w * out_channels)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def _init_and_register_col_mask(self, radius=1, wrap=False):
            # Build once using float32 + CPU, register as buffer, reuse during training/inference
            mask_cpu = build_col_neighbor_mask(
                self.x_embedder.grid_size, device='cpu', dtype=torch.float32, radius=radius, wrap=wrap
            )  # Shape (1,1,T,T)
            # persistent=False: not saved with checkpoint; change to True if you want it saved with checkpoint
            self.register_buffer("col_mask_cpu", mask_cpu, persistent=False)
            # Record parameters (optional)
            self.col_mask_radius = radius
            self.col_mask_wrap = wrap

    def __init__(
        self,
        input_size=(32, 32),
        patch_size=(4,1),             # int -> square; (ph, pw) -> rectangular
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        use_col_mask=False,
        radius=1,
        dict_D=None,                  # (C, d_patch, rank) -- global PCA dictionary
        dict_mean=None,               # (C, d_patch) -- per-channel patch mean
        use_dict_cond=True,           # whether to enable cross-attention to dict tokens
        img_patch_grid=None,          # (Nph, Npw): patch-space grid size of the original image, e.g. (16, 16)
                                      # If set, positional encoding uses these 2D coordinates instead of the DiT patch-grid coordinates
        channel_wise_embed=False,     # If True, use ChannelWisePatchEmbed instead of channel-mixing PatchEmbed
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.use_col_mask = use_col_mask
        self.radius = radius
        self.num_classes = num_classes
        self.img_patch_grid = tuple(img_patch_grid) if img_patch_grid is not None else None

        # Register dictionary as non-parameter buffers (saved in checkpoint, moved with .to())
        use_dict_cond = (dict_D is not None) and use_dict_cond
        self.use_dict_cond = use_dict_cond
        self.register_buffer('dict_D',    dict_D.float()    if dict_D    is not None else None)
        self.register_buffer('dict_mean', dict_mean.float() if dict_mean is not None else None)

        # ---- helpers to normalize sizes ----
        def _to_hw(val, name):
            if isinstance(val, int):
                return (val, val)
            if isinstance(val, (tuple, list)) and len(val) == 2:
                return (int(val[0]), int(val[1]))
            raise ValueError(f"{name} must be int or tuple/list of length 2, got: {val!r}")

        self.input_size = _to_hw(input_size, "input_size")
        self.patch_size = _to_hw(patch_size, "patch_size")   # (ph, pw)

        # ---- embedders ----
        if channel_wise_embed:
            self.x_embedder = ChannelWisePatchEmbed(
                self.input_size, self.patch_size, in_channels, hidden_size, bias=True
            )
        else:
            self.x_embedder = PatchEmbed(
                self.input_size, self.patch_size, in_channels, hidden_size, bias=True
            )
        self.t_embedder = TimestepEmbedder(hidden_size)
        if num_classes > 0:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            self.null_y = nn.Parameter(torch.zeros(hidden_size))
            nn.init.normal_(self.null_y, std=0.02)
        else:
            self.y_embedder = None
            self.register_parameter("null_y", None)

        num_patches = self.x_embedder.num_patches   # Gh * Gw
        if self.img_patch_grid is not None:
            assert self.img_patch_grid[0] * self.img_patch_grid[1] == num_patches, (
                f"img_patch_grid {self.img_patch_grid} total size "
                f"{self.img_patch_grid[0]*self.img_patch_grid[1]} != num_patches {num_patches}"
            )
        # Fixed sin-cos pos embed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     use_dict_cond=use_dict_cond) for _ in range(depth)
        ])
        # FinalLayer we wrote earlier already supports int/tuple; pass the tuple here:
        self.final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels)

        # Projection from patch dimension to hidden_size for dictionary tokens
        if use_dict_cond:
            d_patch = dict_D.shape[1]   # e.g. 64 for 8x8 patches
            self.dict_proj = nn.Linear(d_patch, hidden_size, bias=True)
        else:
            self.dict_proj = None

        self.initialize_weights()
        if self.use_col_mask:
            print(f"Using column mask with radius {self.radius}")
            self._init_and_register_col_mask(radius=self.radius, wrap=False)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        _pos_grid_size = self.img_patch_grid if self.img_patch_grid is not None \
                         else self.x_embedder.grid_size
        pos_embed = get_2d_sincos_pos_embed_from_grid(
            self.pos_embed.shape[-1],
            tuple_to_grid(_pos_grid_size)   # (2, num_patches)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if isinstance(self.x_embedder, ChannelWisePatchEmbed):
            for linear in self.x_embedder.projs:
                nn.init.xavier_uniform_(linear.weight)
                if linear.bias is not None:
                    nn.init.constant_(linear.bias, 0)
            nn.init.ones_(self.x_embedder.channel_scale)
        else:
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.y_embedder is not None:
            if hasattr(self.y_embedder, "embedding_table"):
                nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
            elif hasattr(self.y_embedder, "emb"):
                nn.init.normal_(self.y_embedder.emb.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            # Zero-init cross-attention output projection so it starts as identity pass-through
            if block.use_dict_cond:
                nn.init.constant_(block.cross_proj.weight, 0)
                nn.init.constant_(block.cross_proj.bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, grid_size):
        """
        x: (B, T, ph*pw*C)
        imgs: (B, C, H, W)
        grid_size: (Gh, Gw)
        """
        c = self.out_channels
        ph, pw = self.x_embedder.patch_size   # (ph, pw)
        Gh, Gw = grid_size
        assert Gh * Gw == x.shape[1], f"grid {Gh}x{Gw} != tokens {x.shape[1]}"

        # reshape tokens -> grid of patches -> image
        B = x.shape[0]
        x = x.reshape(B, Gh, Gw, c, ph, pw)                  # (B,Gh,Gw,C,ph,pw)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()         # (B,C,Gh,ph,Gw,pw)
        imgs = x.view(B, c, Gh * ph, Gw * pw)                # (B,C,H,W)
        return imgs

    def forward(self, x, t, y=None):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,) or None
        """
        x = self.x_embedder(x) + self.pos_embed     # (N, T=Hp*Wp, D)
        t = self.t_embedder(t)                      # (N, D)

        # (Tiny logic fix: if labels provided, use them; else dropout-to-null)
        if self.num_classes > 0:
            if y is None:
                y_emb = self.null_y.to(t.device, dtype=t.dtype).expand_as(t)
            else:
                y = y.to(t.device).long()
                y_emb = self.y_embedder(y, self.training)   # (N, D)
            c = t + y_emb
        else:
            c = t

        # Encode global dictionary into cross-attention tokens (once per forward pass)
        # dict_D: (C, d_patch, rank) -> flatten to (C*rank, d_patch) -> project to (1, C*rank, hidden)
        if self.use_dict_cond and self.dict_D is not None:
            C, d_patch, rank = self.dict_D.shape
            d_flat = self.dict_D.permute(0, 2, 1).reshape(C * rank, d_patch)  # (C*rank, d_patch)
            dict_tokens = self.dict_proj(d_flat).unsqueeze(0)                  # (1, C*rank, hidden)
        else:
            dict_tokens = None

        # Attention mask for same/adjacent columns only
        if self.use_col_mask:
            # Fallback: if not yet registered (e.g., switch turned from False to True at runtime), register now
            if not hasattr(self, "col_mask_cpu"):
                self._init_and_register_col_mask(radius=self.radius, wrap=False)
            attn_mask = self.col_mask_cpu.to(device=x.device)  # Keep float32 for better stability
        else:
            attn_mask = None  # No mask

        for block in self.blocks:
            x = block(x, c, attn_mask=attn_mask, dict_tokens=dict_tokens)

        x = self.final_layer(x, c)
        x = self.unpatchify(x, self.x_embedder.grid_size)
        return x



       

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

#################################################################################
#                   JointDiT -- Joint Alpha + Dict Diffusion (Choice B)         #
#################################################################################
#################################################################################
#                   JointDiT -- Joint Alpha + V_hat Diffusion                   #
#################################################################################

class AlphaDictEmbedder(nn.Module):
    """
    Embeds a joint tensor x of shape (B, 3, alpha_n + patch_dim, rank), where

      - x[:, :, :alpha_n, :]      = alpha   with shape (B, 3, alpha_n, rank)
      - x[:, :, alpha_n:, :]      = V_hat   with shape (B, 3, patch_dim, rank)

    Tokenization:
      - alpha part -> alpha_n tokens, each token dim = 3 * rank
      - V_hat part -> rank tokens, each token dim = 3 * patch_dim

    For 1024x1024 image with patch=32, rank=16:
      alpha_n   = (1024/32)^2 = 16384
      patch_dim = 32*32 = 64
      joint     = (B, 3, 16448, 16)
    """
    def __init__(self, hidden_size, alpha_n, rank=16, patch_dim=64):
        super().__init__()
        self.alpha_n = alpha_n
        self.rank = rank
        self.patch_dim = patch_dim

        self.alpha_proj = nn.Linear(3 * rank, hidden_size, bias=True)
        self.dict_proj = nn.Linear(3 * patch_dim, hidden_size, bias=True)
        self.dict_rank_embed = nn.Embedding(rank, hidden_size)

        self.register_buffer(
            "pos_embed",
            torch.zeros(1, alpha_n, hidden_size),
            persistent=False
        )

    def forward(self, x):
        # x: (B, 3, alpha_n + patch_dim, rank)
        B, C, H, R = x.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert R == self.rank, f"Expected rank={self.rank}, got {R}"
        assert H == self.alpha_n + self.patch_dim, (
            f"Expected joint height {self.alpha_n + self.patch_dim}, got {H}"
        )

        x_alpha = x[:, :, :self.alpha_n, :]      # (B, 3, alpha_n, rank)
        x_dict  = x[:, :, self.alpha_n:, :]      # (B, 3, patch_dim, rank)

        # Alpha embed:
        # (B,3,alpha_n,rank) -> (B,alpha_n,3*rank)
        xa = x_alpha.permute(0, 2, 1, 3).reshape(B, self.alpha_n, 3 * self.rank)
        xa = self.alpha_proj(xa) + self.pos_embed

        # V_hat embed:
        # x_dict: (B,3,patch_dim,rank)
        # -> (B,3,rank,patch_dim)
        # -> (B,rank,3,patch_dim)
        # -> (B,rank,3*patch_dim)
        xd = x_dict.permute(0, 1, 3, 2) \
                   .permute(0, 2, 1, 3) \
                   .reshape(B, self.rank, 3 * self.patch_dim)

        rank_ids = torch.arange(self.rank, device=x.device)
        xd = self.dict_proj(xd) + self.dict_rank_embed(rank_ids)

        return torch.cat([xa, xd], dim=1)   # (B, alpha_n + rank, hidden)


class JointFinalLayer(nn.Module):
    """
    Final layer for JointDiT:
      - alpha tokens -> (B, 3, alpha_n, rank)
      - V_hat tokens -> (B, 3, patch_dim, rank)
    """
    def __init__(self, hidden_size, alpha_n, rank=16, patch_dim=64):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.alpha_head = nn.Linear(hidden_size, 3 * rank, bias=True)
        self.dict_head  = nn.Linear(hidden_size, 3 * patch_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        self.alpha_n = alpha_n
        self.rank = rank
        self.patch_dim = patch_dim

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)

        B = x.shape[0]
        total_tokens = x.shape[1]
        expected_tokens = self.alpha_n + self.rank
        assert total_tokens == expected_tokens, (
            f"Expected {expected_tokens} tokens, got {total_tokens}"
        )

        xa = x[:, :self.alpha_n, :]   # (B, alpha_n, hidden)
        xd = x[:, self.alpha_n:, :]   # (B, rank, hidden)

        # Alpha:
        # (B, alpha_n, 3*rank) -> (B, alpha_n, 3, rank) -> (B, 3, alpha_n, rank)
        alpha = (
            self.alpha_head(xa)
            .reshape(B, self.alpha_n, 3, self.rank)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # V_hat:
        # (B, rank, 3*patch_dim)
        # -> (B, rank, 3, patch_dim)
        # -> (B, 3, rank, patch_dim)
        # -> (B, 3, patch_dim, rank)
        d = (
            self.dict_head(xd)
            .reshape(B, self.rank, 3, self.patch_dim)
            .permute(0, 2, 1, 3)
            .permute(0, 1, 3, 2)
            .contiguous()
        )

        return alpha, d


class JointDiT(nn.Module):
    """
    Joint diffusion over alpha and V_hat.

    Input:
      x: (B, 3, alpha_n + patch_dim, rank)

    where
      alpha_n   = (img_size // patch_size)^2
      patch_dim = patch_size^2

    Example:
      img_size   = 1024
      patch_size = 32
      rank       = 16

      alpha_n    = 1024
      patch_dim  = 1024
      joint      = (B, 3, 2048, 16)

    Tokenization:
      - alpha_n alpha tokens
      - rank V_hat tokens
    """
    def __init__(
        self,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        img_size=1024,
        patch_size=32,
        rank=16,
    ):
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")

        self.img_size = img_size
        self.patch_size = patch_size
        self.rank = rank
        self.patch_dim = patch_size * patch_size
        self.grid_size = img_size // patch_size
        self.alpha_n = self.grid_size * self.grid_size
        self.joint_h = self.alpha_n + self.patch_dim

        self.embedder = AlphaDictEmbedder(
            hidden_size=hidden_size,
            alpha_n=self.alpha_n,
            rank=self.rank,
            patch_dim=self.patch_dim,
        )
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_dict_cond=False)
            for _ in range(depth)
        ])

        self.final_layer = JointFinalLayer(
            hidden_size=hidden_size,
            alpha_n=self.alpha_n,
            rank=self.rank,
            patch_dim=self.patch_dim,
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Positional embedding for alpha tokens:
        # patch grid = (img_size // patch_size, img_size // patch_size)
        pos_embed = get_2d_sincos_pos_embed_from_grid(
            self.embedder.pos_embed.shape[-1],
            tuple_to_grid((self.grid_size, self.grid_size))
        )
        self.embedder.pos_embed.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.embedder.dict_rank_embed.weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.alpha_head.weight, 0)
        nn.init.constant_(self.final_layer.alpha_head.bias, 0)
        nn.init.constant_(self.final_layer.dict_head.weight, 0)
        nn.init.constant_(self.final_layer.dict_head.bias, 0)

    def forward(self, x, t, y=None):
        """
        x: (B, 3, joint_h, rank) where joint_h = alpha_n + patch_dim
        t: (B,)
        Returns: (B, 3, joint_h, rank)
        """
        B, C, H, R = x.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert H == self.joint_h, f"Expected joint_h={self.joint_h}, got {H}"
        assert R == self.rank, f"Expected rank={self.rank}, got {R}"

        tokens = self.embedder(x)   # (B, alpha_n + rank, hidden)
        c = self.t_embedder(t)      # (B, hidden)

        for block in self.blocks:
            tokens = block(tokens, c)

        alpha_hat, d_hat = self.final_layer(tokens, c)
        return torch.cat([alpha_hat, d_hat], dim=2)   # (B, 3, joint_h, rank)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#           USVDiT -- Joint U / Sigma / V Three-Part Diffusion                       #
#   Input data: U_hat (B,3,256,R) | S_mid (B,3,R,R) | V_hat (B,3,64,R)        #
#   Packed tensor: (B,3,336,16) = [U(256) | S_T(16) | V(64)] along dim=2       #
#   S stored transposed: row k of S_T = column k of S_mid                       #
#   Tokens: 256 U + 16 Sigma + 16 V = 288 tokens                                   #
#################################################################################

class USVEmbedder(nn.Module):
    """
    Embed packed (B, 3, 336, 16) -> (B, 288, hidden_size).

    Packing convention:
      data[:, :,   0:256, :] = U_hat_norm          (B, 3, 256, 16)
      data[:, :, 256:272, :] = S_mid_norm^T         (B, 3,  16, 16)  row k = column k of S_mid
      data[:, :, 272:336, :] = V_hat_norm           (B, 3,  64, 16)

    Token types (type_embed index):
      0 = U tokens  (256, 48-dim  -> hidden)  + sincos pos_embed
      1 = Sigma tokens  ( 16, 48-dim  -> hidden)  + rank_embed (shared with V)
      2 = V tokens  ( 16, 192-dim -> hidden)  + rank_embed (shared with Sigma)
    """
    def __init__(self, hidden_size: int, num_rank_tokens: int = 16,
                 spatialize_s: bool = False):
        super().__init__()
        self.u_proj     = nn.Linear(3 * 16, hidden_size, bias=True)   # 48  -> hidden
        self.sigma_proj = nn.Linear(3 * 16, hidden_size, bias=True)   # 48  -> hidden
        self.v_proj     = nn.Linear(3 * 64, hidden_size, bias=True)   # 192 -> hidden
        # rank embedding shared between Sigma tokens and V tokens
        self.rank_embed = nn.Embedding(num_rank_tokens, hidden_size)
        # token type embedding: 0=U, 1=Sigma, 2=V
        self.type_embed = nn.Embedding(3, hidden_size)
        # sincos pos_embed for U tokens -- filled by USVDiT.initialize_weights
        self.register_buffer('pos_embed', torch.zeros(1, 256, hidden_size), persistent=False)
        # optional per-rank attention pooling from U -> S
        self.spatialize_s = spatialize_s
        if spatialize_s:
            self.s_pool_q    = nn.Parameter(torch.randn(num_rank_tokens, hidden_size) * 0.02)
            self.s_pool_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 336, 16)
        B = x.shape[0]
        u_raw    = x[:, :,   0:256, :]   # (B, 3, 256, 16)
        s_packed = x[:, :, 256:272, :]   # (B, 3,  16, 16)  row k = column k of S_mid
        v_raw    = x[:, :, 272:336, :]   # (B, 3,  64, 16)

        rank_ids = torch.arange(16, device=x.device)

        # -- U tokens ---------------------------------------------------------
        # (B,3,256,16) -> permute(0,2,1,3) -> (B,256,3,16) -> reshape -> (B,256,48)
        u_feat = u_raw.permute(0, 2, 1, 3).reshape(B, 256, 3 * 16)   # (B, 256, 48)
        u_tok  = (self.u_proj(u_feat)                                  # (B, 256, hidden)
                  + self.pos_embed                                     # sincos spatial
                  + self.type_embed.weight[0])                         # type 0 = U

        # -- Sigma tokens ---------------------------------------------------------
        # s_packed: (B,3,16,16) where row k = column k of S_mid
        # -> permute(0,2,1,3) -> (B,16,3,16) -> reshape -> (B,16,48)
        s_feat = s_packed.permute(0, 2, 1, 3).reshape(B, 16, 3 * 16)  # (B, 16, 48)
        s_tok  = (self.sigma_proj(s_feat)                               # (B, 16, hidden)
                  + self.rank_embed(rank_ids)                           # shared rank embed
                  + self.type_embed.weight[1])                          # type 1 = Sigma

        # -- optional: inject spatial context from U into each S_k ---------
        if self.spatialize_s:
            # per-rank attention pooling: s_pool_q (R, H) queries over u_tok (B, 256, H)
            scale = self.s_pool_q.shape[-1] ** -0.5
            q    = self.s_pool_q.unsqueeze(0).expand(B, -1, -1)    # (B, R, H)
            attn = torch.bmm(q, u_tok.transpose(1, 2)) * scale     # (B, R, 256)
            attn = attn.softmax(dim=-1)
            ctx  = torch.bmm(attn, u_tok)                           # (B, R, H)
            s_tok = s_tok + self.s_pool_proj(ctx)

        # -- V tokens ---------------------------------------------------------
        # (B,3,64,16) -> permute(0,1,3,2) -> (B,3,16,64) -> permute(0,2,1,3)
        # -> (B,16,3,64) -> reshape -> (B,16,192)
        v_feat = v_raw.permute(0, 1, 3, 2).permute(0, 2, 1, 3).reshape(B, 16, 3 * 64)
        v_tok  = (self.v_proj(v_feat)                                   # (B, 16, hidden)
                  + self.rank_embed(rank_ids)                           # shared rank embed
                  + self.type_embed.weight[2])                          # type 2 = V

        return torch.cat([u_tok, s_tok, v_tok], dim=1)   # (B, 288, hidden)


class USVFinalLayer(nn.Module):
    """
    Three separate output heads for U / Sigma / V tokens.
    Reconstructs packed (B, 3, 336, 16) -- same format as input to USVDiT.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.u_head     = nn.Linear(hidden_size, 3 * 16, bias=True)   # hidden -> 48
        self.sigma_head = nn.Linear(hidden_size, 3 * 16, bias=True)   # hidden -> 48
        self.v_head     = nn.Linear(hidden_size, 3 * 64, bias=True)   # hidden -> 192
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, 288, hidden),  c: (B, hidden)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        B = x.shape[0]

        xu = x[:, 0:256, :]    # (B, 256, hidden)
        xs = x[:, 256:272, :]  # (B,  16, hidden)
        xv = x[:, 272:288, :]  # (B,  16, hidden)

        # U head: (B,256,48) -> reshape(B,256,3,16) -> permute(0,2,1,3) -> (B,3,256,16)
        u_out = self.u_head(xu).reshape(B, 256, 3, 16).permute(0, 2, 1, 3)   # (B,3,256,16)

        # Sigma head: (B,16,48) -> reshape(B,16,3,16) -> permute(0,2,1,3) -> (B,3,16,16) = S_packed
        #   S_packed stores row k = column k of S_mid; keep this format for packing into data tensor
        s_out = self.sigma_head(xs).reshape(B, 16, 3, 16).permute(0, 2, 1, 3)   # (B,3,16,16)

        # V head: (B,16,192) -> reshape(B,16,3,64) -> permute(0,2,1,3) -> (B,3,16,64)
        #         -> permute(0,1,3,2) -> (B,3,64,16)
        v_out = (self.v_head(xv)
                 .reshape(B, 16, 3, 64)
                 .permute(0, 2, 1, 3)    # (B, 3, 16, 64)
                 .permute(0, 1, 3, 2))   # (B, 3, 64, 16)

        # Pack back into (B, 3, 336, 16): S is already in packed (transposed) format
        return torch.cat([u_out, s_out, v_out], dim=2)   # (B, 3, 336, 16)


class USVDiT(nn.Module):
    """
    Jointly diffuse U_hat / S_mid / V_hat in the USV decomposition.

    Input:  (B, 3, 336, 16) -- packed [U(256) | S_T(16) | V(64)]
    Tokens: 256 U + 16 Sigma + 16 V = 288 total (full global self-attention)
    Output: (B, 3, 336, 16) -- predicted noise in packed format
    """
    def __init__(self, hidden_size: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 spatialize_s: bool = False):
        super().__init__()
        self.embedder    = USVEmbedder(hidden_size, spatialize_s=spatialize_s)
        self.t_embedder  = TimestepEmbedder(hidden_size)
        self.blocks      = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_dict_cond=False)
            for _ in range(depth)
        ])
        self.final_layer = USVFinalLayer(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # 1. Xavier uniform for all Linear layers, zero bias
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 2. Sincos pos_embed for U tokens -- 16x16 patch grid (128px / 8px patch)
        pos_embed = get_2d_sincos_pos_embed_from_grid(
            self.embedder.pos_embed.shape[-1],
            tuple_to_grid((16, 16))
        )
        self.embedder.pos_embed.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 3. Normal init for learnable embeddings
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.embedder.rank_embed.weight, std=0.02)
        nn.init.normal_(self.embedder.type_embed.weight, std=0.02)
        if self.embedder.spatialize_s:
            nn.init.normal_(self.embedder.s_pool_q, std=0.02)
            nn.init.constant_(self.embedder.s_pool_proj.weight, 0)

        # 4. adaLN-Zero: zero-init last Linear in each block's modulation
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias,   0)

        # 5. Zero-init final layer heads and modulation (adaLN-Zero output gate)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.final_layer.u_head.weight,     0)
        nn.init.constant_(self.final_layer.u_head.bias,       0)
        nn.init.constant_(self.final_layer.sigma_head.weight, 0)
        nn.init.constant_(self.final_layer.sigma_head.bias,   0)
        nn.init.constant_(self.final_layer.v_head.weight,     0)
        nn.init.constant_(self.final_layer.v_head.bias,       0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None) -> torch.Tensor:
        """
        x: (B, 3, 336, 16) -- packed [U_norm | S_norm_T | V_norm]
        t: (B,) -- diffusion timesteps
        Returns: (B, 3, 336, 16) -- predicted noise in same packed format
        """
        tokens = self.embedder(x)        # (B, 288, hidden)
        c      = self.t_embedder(t)      # (B, hidden)
        for block in self.blocks:
            tokens = block(tokens, c)
        return self.final_layer(tokens, c)   # (B, 3, 336, 16)


