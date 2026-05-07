"""U-ViT variant for PDE conditional generation.

Differences vs. libs/uvit.py:
  * Conditions on a continuous scalar nu (sinusoidal embedding -> 2-layer MLP)
    instead of an integer class label via nn.Embedding.
  * Otherwise identical layout: linear-project DCT-token features into the
    embed_dim, prepend [nu_token, time_token], run U-shaped transformer,
    decode back to the per-token feature dim.
"""

import math
import torch
import torch.nn as nn

from .timm import trunc_normal_
from .uvit import Block, timestep_embedding


class UViTPDE(nn.Module):
    def __init__(
        self,
        tokens, low_freqs, M_x=4,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
        qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm,
        mlp_time_embed=False, use_checkpoint=False, conv=True, skip=True,
        nu_log_scale=True, nu_log_min=-5.0, nu_log_max=-1.0,
        **_,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.tokens = tokens
        self.feature_dim = M_x * low_freqs
        self.nu_log_scale = nu_log_scale
        self.nu_log_min = nu_log_min
        self.nu_log_max = nu_log_max

        self.proj = nn.Linear(self.feature_dim, embed_dim, bias=True)

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        # nu conditioning: scalar -> sinusoidal embed -> small MLP -> token.
        self.nu_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        # Two extras: nu_token + time_token.
        self.extras = 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + tokens, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
                  use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)
        ])
        self.mid_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.out_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
                  skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)
        ])

        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, self.feature_dim, bias=True)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def _nu_to_token(self, nu):
        # Map nu (positive scalar, log-spaced 1e-5 .. 1e-1) into a roughly
        # standardized scalar before sinusoidal embedding.
        if self.nu_log_scale:
            s = torch.log10(nu.clamp(min=1e-12))
            s = (s - self.nu_log_min) / (self.nu_log_max - self.nu_log_min)
            s = s * 1000.0
        else:
            s = nu * 1000.0
        emb = timestep_embedding(s, self.embed_dim)
        return self.nu_embed(emb)

    def forward(self, x, timesteps, nu=None):
        # x: (B, tokens, feature_dim)
        x = self.proj(x)                                          # (B, L, D)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(1)                      # (B, 1, D)

        nu_token = self._nu_to_token(nu).unsqueeze(1)             # (B, 1, D)

        x = torch.cat((nu_token, time_token, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)
        x = self.mid_block(x)
        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        x = x[:, self.extras:, :]                                 # drop nu/time tokens
        return x


class UViTReaction(nn.Module):
    """U-ViT for the 1D reaction-diffusion (Fisher-KPP) PDE.

    Conditions on TWO continuous scalars:
      nu  : viscosity, log-spaced in [1e-5, 1e-1]
      rho : reaction rate, log-spaced in [1e-1, 2.0]

    Each is sinusoidally embedded and projected by its own 2-layer MLP into a
    conditioning token. The transformer stack is identical to UViTPDE except
    for one extra prepended token (rho_token).
    """

    def __init__(
        self,
        tokens, low_freqs, M_x=4,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
        qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm,
        mlp_time_embed=False, use_checkpoint=False, conv=True, skip=True,
        nu_log_scale=True, nu_log_min=-5.0, nu_log_max=-1.0,
        rho_log_scale=True, rho_log_min=-1.0, rho_log_max=0.301,
        **_,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.tokens = tokens
        self.feature_dim = M_x * low_freqs
        self.nu_log_scale = nu_log_scale
        self.nu_log_min = nu_log_min
        self.nu_log_max = nu_log_max
        self.rho_log_scale = rho_log_scale
        self.rho_log_min = rho_log_min
        self.rho_log_max = rho_log_max

        self.proj = nn.Linear(self.feature_dim, embed_dim, bias=True)

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.nu_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.rho_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        # Three extras: nu_token + rho_token + time_token.
        self.extras = 3
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + tokens, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
                  use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)
        ])
        self.mid_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.out_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
                  skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)
        ])

        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, self.feature_dim, bias=True)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def _scalar_to_token(self, val, embed, log_scale, log_min, log_max):
        if log_scale:
            s = torch.log10(val.clamp(min=1e-12))
            s = (s - log_min) / (log_max - log_min)
            s = s * 1000.0
        else:
            s = val * 1000.0
        emb = timestep_embedding(s, self.embed_dim)
        return embed(emb)

    def forward(self, x, timesteps, nu=None, rho=None):
        # x: (B, tokens, feature_dim)
        x = self.proj(x)                                          # (B, L, D)

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(1)                      # (B, 1, D)

        nu_token = self._scalar_to_token(
            nu, self.nu_embed, self.nu_log_scale,
            self.nu_log_min, self.nu_log_max,
        ).unsqueeze(1)                                            # (B, 1, D)
        rho_token = self._scalar_to_token(
            rho, self.rho_embed, self.rho_log_scale,
            self.rho_log_min, self.rho_log_max,
        ).unsqueeze(1)                                            # (B, 1, D)

        x = torch.cat((nu_token, rho_token, time_token, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)
        x = self.mid_block(x)
        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        x = x[:, self.extras:, :]                                 # drop nu/rho/time tokens
        return x
