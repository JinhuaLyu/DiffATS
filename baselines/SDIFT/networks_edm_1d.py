from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import silu


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f"invalid init mode '{mode}'")


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 init_mode="kaiming_normal", init_weight=1, init_bias=0):
        super().__init__()
        kw = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = nn.Parameter(weight_init([out_features, in_features], **kw) * init_weight)
        self.bias = nn.Parameter(weight_init([out_features], **kw) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x + self.bias.to(x.dtype)
        return x


class Conv1dCustom(nn.Module):
    """Conv1d with optional symmetric up/downsampling (factor 2)."""

    def __init__(self, in_channels, out_channels, kernel, bias=True, up=False, down=False,
                 resample_filter=(1, 1), init_mode="kaiming_normal",
                 init_weight=1, init_bias=0):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        kw = dict(mode=init_mode, fan_in=in_channels * max(kernel, 1),
                  fan_out=out_channels * max(kernel, 1))
        self.weight = (
            nn.Parameter(weight_init([out_channels, in_channels, kernel], **kw) * init_weight)
            if kernel else None
        )
        self.bias = (
            nn.Parameter(weight_init([out_channels], **kw) * init_bias)
            if (kernel and bias) else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        if f.sum() > 0:
            f = (f / f.sum()).unsqueeze(0).unsqueeze(0)  # (1,1,k)
        else:
            f = f.unsqueeze(0).unsqueeze(0)
        self.register_buffer("resample_filter", f if (up or down) else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = (w.shape[-1] // 2) if w is not None else 0
        f_pad = ((f.shape[-1] - 1) // 2) if f is not None else 0

        if self.up:
            x = torch.nn.functional.conv_transpose1d(
                x, f.mul(2).tile([self.in_channels, 1, 1]),
                groups=self.in_channels, stride=2, padding=f_pad,
            )
        if self.down:
            x = torch.nn.functional.conv1d(
                x, f.tile([self.in_channels, 1, 1]),
                groups=self.in_channels, stride=2, padding=f_pad,
            )
        if w is not None:
            x = torch.nn.functional.conv1d(x, w, padding=w_pad)
        if b is not None:
            x = x + b.reshape(1, -1, 1)
        return x


class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        # Largest divisor of num_channels that is <= num_groups and yields
        # >= min_channels_per_group channels per group.
        max_groups = min(num_groups, max(num_channels // min_channels_per_group, 1))
        ng = 1
        for g in range(max_groups, 0, -1):
            if num_channels % g == 0:
                ng = g
                break
        self.num_groups = ng
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return torch.nn.functional.group_norm(
            x, num_groups=self.num_groups,
            weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype),
            eps=self.eps,
        )


class UNetBlock1D(nn.Module):
    """Residual U-Net block with embedding-conditioning and optional attention."""

    def __init__(self, in_channels, out_channels, emb_channels, up=False, down=False,
                 attention=False, num_heads=None, channels_per_head=64, dropout=0.0,
                 skip_scale=1.0, eps=1e-5, resample_filter=(1, 1), resample_proj=False,
                 adaptive_scale=True, init=None, init_zero=None, init_attn=None):
        super().__init__()
        init = init or {}
        init_zero = init_zero or dict(init_weight=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0 if not attention
            else (num_heads if num_heads is not None else max(out_channels // channels_per_head, 1))
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv1dCustom(in_channels, out_channels, kernel=3, up=up, down=down,
                                  resample_filter=resample_filter, **init)
        self.affine = Linear(emb_channels, out_channels * (2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv1dCustom(out_channels, out_channels, kernel=3, **init_zero)

        self.skip = None
        if (out_channels != in_channels) or up or down:
            kernel = 1 if (resample_proj or out_channels != in_channels) else 0
            self.skip = Conv1dCustom(in_channels, out_channels, kernel=kernel, up=up, down=down,
                                     resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv1dCustom(out_channels, out_channels * 3, kernel=1,
                                    **(init_attn if init_attn is not None else init))
            self.proj = Conv1dCustom(out_channels, out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x + params))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x + (self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            qkv = self.qkv(self.norm2(x))
            B, _, L = qkv.shape
            qkv = qkv.reshape(B * self.num_heads, x.shape[1] // self.num_heads, 3, -1)
            q, k, v = qkv.unbind(2)  # each (B*heads, c_per_head, L)
            w = torch.einsum("ncq,nck->nqk", q.float(),
                              (k / np.sqrt(k.shape[1])).float()).softmax(dim=2).to(q.dtype)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)) + x
            x = x * self.skip_scale
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(0, self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class FourierEmbedding(nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


# Main model.
class Spatial_temporal_UNet_1D(nn.Module):
    """1D spatial U-Net + per-block temporal Conv1d, with optional conditioning.

    Inputs to ``forward``:
      * x            : (B, T, C, R1)            — Tucker cores.
      * noise_labels : (B, T, 1)                — log-sigma per timestep (broadcast from per-batch sigma).
      * time_labels  : (B, T, 1)                — physical time indices in [0, 1].
      * cond         : (B, cond_dim) or None    — per-sample conditioning vector.
    """

    def __init__(self,
                 r1_resolution: int,
                 in_channels: int,
                 out_channels: int,
                 model_channels: int = 64,
                 channel_mult: List[int] = (1, 2, 2, 2),
                 channel_mult_emb: int = 4,
                 num_blocks: int = 2,
                 num_temporal_latent: int = 4,
                 attn_resolutions: List[int] = (),
                 dropout: float = 0.0,
                 embedding_type: str = "positional",
                 channel_mult_noise: int = 1,
                 cond_dim: int = 0):
        super().__init__()
        assert embedding_type in ("positional", "fourier")
        self.r1_resolution = r1_resolution
        self.cond_dim = cond_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout,
            skip_scale=np.sqrt(0.5), eps=1e-6, resample_filter=(1, 1),
            resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        self.map_noise = (
            PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            if embedding_type == "positional"
            else FourierEmbedding(num_channels=noise_channels)
        )
        self.map_time = FourierEmbedding(num_channels=noise_channels)
        self.map = nn.Sequential(
            nn.Linear(noise_channels * 2, emb_channels),
            nn.ReLU(),
            nn.Linear(emb_channels, emb_channels),
        )

        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, emb_channels),
                nn.SiLU(),
                nn.Linear(emb_channels, emb_channels),
            )
            # Initialise final layer near zero so at start the conditioning has minimal effect.
            with torch.no_grad():
                self.cond_mlp[-1].weight.mul_(1e-2)
                self.cond_mlp[-1].bias.zero_()
        else:
            self.cond_mlp = None

        # Encoder.
        self.enc = nn.ModuleDict()
        self.enc_temp = nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = r1_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}_conv"] = Conv1dCustom(cin, cout, kernel=3, **init)
                self.enc_temp[f"{res}_conv"] = nn.Sequential(
                    nn.Conv1d(cout, num_temporal_latent * cout, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(num_temporal_latent * cout, cout, kernel_size=3, stride=1, padding=1),
                )
            else:
                self.enc[f"{res}_down"] = UNetBlock1D(cout, cout, down=True, **block_kwargs)
                self.enc_temp[f"{res}_down"] = nn.Sequential(
                    nn.Conv1d(cout, num_temporal_latent * cout, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(num_temporal_latent * cout, cout, kernel_size=3, stride=1, padding=1),
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}_block{idx}"] = UNetBlock1D(cin, cout, attention=attn, **block_kwargs)
                self.enc_temp[f"{res}_block{idx}"] = nn.Sequential(
                    nn.Conv1d(cout, num_temporal_latent * cout, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(num_temporal_latent * cout, cout, kernel_size=3, stride=1, padding=1),
                )
        skips = [b.out_channels for b in self.enc.values()]

        # Decoder.
        self.dec = nn.ModuleDict()
        self.dec_temp = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = r1_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}_in0"] = UNetBlock1D(cout, cout, attention=True, **block_kwargs)
                self.dec[f"{res}_in1"] = UNetBlock1D(cout, cout, **block_kwargs)
                self.dec_temp[f"{res}_in0"] = nn.Sequential(
                    nn.Conv1d(cout, num_temporal_latent * cout, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(num_temporal_latent * cout, cout, kernel_size=3, stride=1, padding=1),
                )
                self.dec_temp[f"{res}_in1"] = nn.Sequential(
                    nn.Conv1d(cout, num_temporal_latent * cout, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(num_temporal_latent * cout, cout, kernel_size=3, stride=1, padding=1),
                )
            else:
                self.dec[f"{res}_up"] = UNetBlock1D(cout, cout, up=True, **block_kwargs)
                self.dec_temp[f"{res}_up"] = nn.Sequential(
                    nn.Conv1d(cout, num_temporal_latent * cout, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(num_temporal_latent * cout, cout, kernel_size=3, stride=1, padding=1),
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks) and (res in attn_resolutions)
                self.dec[f"{res}_block{idx}"] = UNetBlock1D(cin, cout, attention=attn, **block_kwargs)
                self.dec_temp[f"{res}_block{idx}"] = nn.Sequential(
                    nn.Conv1d(cout, num_temporal_latent * cout, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(num_temporal_latent * cout, cout, kernel_size=3, stride=1, padding=1),
                )
            if level == 0:
                self.dec[f"{res}_aux_norm"] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f"{res}_aux_conv"] = Conv1dCustom(cout, out_channels, kernel=3, **init_zero)

    # ---- Helpers ---------------------------------------------------------
    @staticmethod
    def _temporal_mix(x_flat, temp_block, B, T):
        """x_flat: (B*T, C, R) -> Conv1d over T at each (b, r) and add as residual.

        Returns the same shape.
        """
        latent_C = x_flat.shape[1]
        latent_R = x_flat.shape[2]
        # (B, T, C, R) -> (B, R, C, T) -> (B*R, C, T)
        temp = x_flat.view(B, T, latent_C, latent_R)
        temp = temp.permute(0, 3, 2, 1).contiguous()
        temp = temp.view(B * latent_R, latent_C, T)
        temp = temp_block(temp)
        # back to (B*T, C, R)
        temp = temp.view(B, latent_R, latent_C, T)
        temp = temp.permute(0, 3, 2, 1).contiguous()
        temp = temp.view(B * T, latent_C, latent_R)
        return temp

    # ---- Forward ---------------------------------------------------------
    def forward(self, x, noise_labels, time_labels, cond=None):
        x_shape = x.shape  # (B, T, C, R1)
        B, T = x_shape[0], x_shape[1]
        x = x.reshape(B * T, *x_shape[2:])  # (B*T, C, R1)
        noise_labels = noise_labels.reshape(-1)  # (B*T,)
        time_labels = time_labels.reshape(-1)    # (B*T,)

        noise_emb = self.map_noise(noise_labels)
        noise_emb = noise_emb.reshape(noise_emb.shape[0], 2, -1).flip(1).reshape(*noise_emb.shape)
        time_emb = self.map_time(time_labels)
        emb = silu(self.map(torch.cat([noise_emb, time_emb], dim=1)))  # (B*T, emb_channels)

        if self.cond_mlp is not None and cond is not None:
            ce = self.cond_mlp(cond)            # (B, emb_channels)
            ce = ce.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            emb = emb + ce

        skips = []
        for name, block in self.enc.items():
            x = block(x, emb) if isinstance(block, UNetBlock1D) else block(x)
            x = x + self._temporal_mix(x, self.enc_temp[name], B, T)
            skips.append(x)

        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
                x = x + self._temporal_mix(x, self.dec_temp[name], B, T)

        return aux.view(x_shape)
