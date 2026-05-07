"""DCT utilities for 1D-PDE conditional generation.

Operates on (T, X) single-channel fields. Tiles into rectangular B_t x B_x
DCT blocks, optionally groups M_x consecutive column-blocks into one token,
and zigzag-orders the per-block coefficients (low to high frequency).

All ops are torch-native (no cv2) so they vectorize on GPU and play well
with autograd if ever needed.
"""

import math
import numpy as np
import torch


def _dct_matrix(N, dtype=torch.float32, device='cpu'):
    """Orthonormal DCT-II matrix of size N x N. y = D @ x."""
    n = torch.arange(N, dtype=dtype, device=device)
    k = n.view(-1, 1)
    D = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
    D[0] *= 1.0 / math.sqrt(N)
    D[1:] *= math.sqrt(2.0 / N)
    return D


class DCT2DBlocks:
    """Apply per-block 2D DCT-II / IDCT-II to a tiled field.

    Shapes:
      field: (..., T, X)
      blocks: (..., n_t, n_x, B_t, B_x)
    """

    def __init__(self, B_t, B_x, device='cpu'):
        self.B_t = B_t
        self.B_x = B_x
        self.D_t = _dct_matrix(B_t, device=device)
        self.D_x = _dct_matrix(B_x, device=device)

    def to(self, device):
        self.D_t = self.D_t.to(device)
        self.D_x = self.D_x.to(device)
        return self

    def split(self, field):
        # field: (..., T, X) -> (..., n_t, n_x, B_t, B_x)
        T, X = field.shape[-2], field.shape[-1]
        assert T % self.B_t == 0 and X % self.B_x == 0, \
            f"field {T}x{X} not divisible by block {self.B_t}x{self.B_x}"
        n_t, n_x = T // self.B_t, X // self.B_x
        out = field.reshape(*field.shape[:-2], n_t, self.B_t, n_x, self.B_x)
        out = out.transpose(-3, -2).contiguous()  # ..., n_t, n_x, B_t, B_x
        return out

    def combine(self, blocks, T=None, X=None):
        # blocks: (..., n_t, n_x, B_t, B_x) -> (..., T, X)
        n_t, n_x = blocks.shape[-4], blocks.shape[-3]
        out = blocks.transpose(-3, -2).contiguous()  # ..., n_t, B_t, n_x, B_x
        out = out.reshape(*blocks.shape[:-4], n_t * self.B_t, n_x * self.B_x)
        if T is not None:
            assert out.shape[-2] == T
        if X is not None:
            assert out.shape[-1] == X
        return out

    def dct(self, blocks):
        # blocks: (..., B_t, B_x) -> same shape, 2D DCT
        D_t = self.D_t.to(blocks.device, blocks.dtype)
        D_x = self.D_x.to(blocks.device, blocks.dtype)
        return torch.einsum('ij,...jk,lk->...il', D_t, blocks, D_x)

    def idct(self, blocks):
        D_t = self.D_t.to(blocks.device, blocks.dtype)
        D_x = self.D_x.to(blocks.device, blocks.dtype)
        return torch.einsum('ji,...jk,kl->...il', D_t, blocks, D_x)


def zigzag_order_2d(B_t, B_x):
    """Indices that reorder a flattened (B_t * B_x) block from row-major
    to zigzag (low->high frequency, anti-diagonal sweep)."""
    out = []
    for s in range(B_t + B_x - 1):
        diag = []
        i_start = max(0, s - (B_x - 1))
        i_end = min(s, B_t - 1)
        for i in range(i_start, i_end + 1):
            j = s - i
            diag.append((i, j))
        if s % 2 == 0:
            diag.reverse()
        out.extend(diag)
    return [i * B_x + j for i, j in out]


def reverse_zigzag_order_2d(B_t, B_x):
    z = zigzag_order_2d(B_t, B_x)
    inv = [0] * (B_t * B_x)
    for new_pos, orig_pos in enumerate(z):
        inv[orig_pos] = new_pos
    return inv


def field_to_tokens(field, B_t, B_x, M_x, low_freqs, zigzag, dct_op=None):
    """Encode a (..., T, X) field into (..., tokens, M_x * low_freqs).

    Token order is row-major over (time-block, x-macro). Within a token, the
    M_x sub-blocks are concatenated and each carries `low_freqs` zigzag-ordered
    DCT coefficients.
    """
    if dct_op is None:
        dct_op = DCT2DBlocks(B_t, B_x).to(field.device)
    blocks = dct_op.split(field)            # (..., n_t, n_x, B_t, B_x)
    coeff = dct_op.dct(blocks)              # same shape
    coeff = coeff.flatten(-2)               # (..., n_t, n_x, B_t*B_x)
    coeff = coeff[..., zigzag]              # zigzag order
    coeff = coeff[..., :low_freqs]          # keep low freqs
    n_t, n_x = coeff.shape[-3], coeff.shape[-2]
    assert n_x % M_x == 0, f"n_x={n_x} not divisible by M_x={M_x}"
    coeff = coeff.reshape(*coeff.shape[:-3], n_t, n_x // M_x, M_x, low_freqs)
    coeff = coeff.reshape(*coeff.shape[:-4], n_t * (n_x // M_x), M_x * low_freqs)
    return coeff


def tokens_to_field(tokens, B_t, B_x, M_x, low_freqs, reverse_zigzag,
                    n_t, n_x, dct_op=None, device=None):
    """Decode (..., tokens, M_x * low_freqs) back to a (..., T, X) field."""
    if dct_op is None:
        dct_op = DCT2DBlocks(B_t, B_x).to(tokens.device)
    n_x_macro = n_x // M_x
    coeff = tokens.reshape(*tokens.shape[:-2], n_t, n_x_macro, M_x, low_freqs)
    coeff = coeff.reshape(*coeff.shape[:-4], n_t, n_x, low_freqs)
    full = torch.zeros(*coeff.shape[:-1], B_t * B_x,
                       dtype=coeff.dtype, device=coeff.device)
    full[..., :low_freqs] = coeff
    inv_idx = torch.tensor(reverse_zigzag, device=full.device)
    full = full[..., inv_idx]              # back to row-major
    full = full.reshape(*full.shape[:-1], B_t, B_x)
    field_blocks = dct_op.idct(full)       # (..., n_t, n_x, B_t, B_x)
    field = dct_op.combine(field_blocks)
    return field


def ic_token_mask(n_t, n_x_macro, ic_block_rows=1):
    """Boolean mask, True where the token belongs to an IC time-block.

    Tokens are ordered (time-block, x-macro). The first `ic_block_rows`
    time-blocks are IC.
    """
    n_tokens = n_t * n_x_macro
    mask = torch.zeros(n_tokens, dtype=torch.bool)
    mask[:ic_block_rows * n_x_macro] = True
    return mask
