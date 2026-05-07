# Alpha-only DiT for shared_bases.
#
# Input:   x of shape (B, 3, alpha_n, rank)   -- only the alpha (projection) part
# Output:  same shape; model predicts alpha only.
# Tokens:  alpha_n tokens, each of dim 3 * rank.
#
# This re-uses the shared DiTBlock / TimestepEmbedder / sincos pos-embed helpers
# from our_method/dit_models.py, but defines its own embedder / final layer
# tailored to alpha-only input (no V_hat tokens).

import os
import sys
import torch
import torch.nn as nn

# Re-use blocks defined in our_method/dit_models.py (read-only, never modified).
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUR_METHOD = os.path.normpath(os.path.join(_HERE, "..", "our_method"))
if _OUR_METHOD not in sys.path:
    sys.path.insert(0, _OUR_METHOD)
from dit_models import (  # noqa: E402
    DiTBlock,
    TimestepEmbedder,
    get_2d_sincos_pos_embed_from_grid,
    tuple_to_grid,
    modulate,
)


class AlphaOnlyEmbedder(nn.Module):
    """
    x: (B, 3, alpha_n, rank) -> tokens (B, alpha_n, hidden_size).
    Each alpha token aggregates the 3-channel rank-r projection of one patch.
    Token feature dim = 3 * rank.
    """
    def __init__(self, hidden_size: int, alpha_n: int, rank: int):
        super().__init__()
        self.alpha_n = alpha_n
        self.rank = rank
        self.alpha_proj = nn.Linear(3 * rank, hidden_size, bias=True)
        self.register_buffer(
            "pos_embed",
            torch.zeros(1, alpha_n, hidden_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N, R = x.shape
        assert C == 3 and N == self.alpha_n and R == self.rank, (
            f"AlphaOnlyEmbedder expected (B,3,{self.alpha_n},{self.rank}); got {tuple(x.shape)}"
        )
        # (B, 3, N, R) -> (B, N, 3, R) -> (B, N, 3*R)
        xa = x.permute(0, 2, 1, 3).reshape(B, N, 3 * R)
        return self.alpha_proj(xa) + self.pos_embed


class AlphaOnlyFinalLayer(nn.Module):
    """
    tokens (B, alpha_n, hidden) -> alpha (B, 3, alpha_n, rank)
    """
    def __init__(self, hidden_size: int, alpha_n: int, rank: int):
        super().__init__()
        self.alpha_n = alpha_n
        self.rank = rank
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(hidden_size, 3 * rank, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)

        B = x.shape[0]
        assert x.shape[1] == self.alpha_n, f"expected {self.alpha_n} tokens, got {x.shape[1]}"
        # (B, alpha_n, 3*rank) -> (B, alpha_n, 3, rank) -> (B, 3, alpha_n, rank)
        return (
            self.head(x)
                 .reshape(B, self.alpha_n, 3, self.rank)
                 .permute(0, 2, 1, 3)
                 .contiguous()
        )


class AlphaOnlyDiT(nn.Module):
    """
    DiT over alpha (the projection coefficients on a global shared basis).
    Same architecture as JointDiT but with alpha_n tokens only.

    For 1024x1024 image with patch=32, rank=32:
      alpha_n = (1024/32)^2 = 1024
      input/output shape = (B, 3, 1024, 32)
    """
    def __init__(
        self,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        img_size: int = 1024,
        patch_size: int = 32,
        rank: int = 32,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")

        self.img_size   = img_size
        self.patch_size = patch_size
        self.rank       = rank
        self.grid_size  = img_size // patch_size
        self.alpha_n    = self.grid_size * self.grid_size

        self.embedder   = AlphaOnlyEmbedder(hidden_size, self.alpha_n, self.rank)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks     = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_dict_cond=False)
            for _ in range(depth)
        ])
        self.final_layer = AlphaOnlyFinalLayer(hidden_size, self.alpha_n, self.rank)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed_from_grid(
            self.embedder.pos_embed.shape[-1],
            tuple_to_grid((self.grid_size, self.grid_size)),
        )
        self.embedder.pos_embed.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.head.weight, 0)
        nn.init.constant_(self.final_layer.head.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None) -> torch.Tensor:
        """
        x: (B, 3, alpha_n, rank)
        t: (B,)
        returns same shape as x
        """
        B, C, N, R = x.shape
        assert C == 3 and N == self.alpha_n and R == self.rank, (
            f"AlphaOnlyDiT expected (B,3,{self.alpha_n},{self.rank}); got {tuple(x.shape)}"
        )

        tokens = self.embedder(x)        # (B, alpha_n, hidden)
        c      = self.t_embedder(t)      # (B, hidden)
        for blk in self.blocks:
            tokens = blk(tokens, c)
        return self.final_layer(tokens, c)   # (B, 3, alpha_n, rank)
