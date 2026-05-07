import os
import torch
from torch.utils.data import Dataset

from DCT_utils_1d import (
    DCT2DBlocks,
    zigzag_order_2d,
    reverse_zigzag_order_2d,
    field_to_tokens,
    ic_token_mask,
)


class Burgers1DDataset(Dataset):
    def __init__(
        self,
        path,
        B_t=8, B_x=10, M_x=4, low_freqs=10,
        T_pad=240, ic_repeat=8, gen_start=40, ic_block_rows=5,
        u_scale=0.6342716217041016,   # std of u over the dataset
        Y_bound=2.6,                  # legacy: scalar fallback if no per_freq_std
        per_freq_std=None,            # if set: per-coefficient std vector for whitening
        train=True,
    ):
        super().__init__()
        # Mmap for the 8 GB train file; safe for read-only access from
        # multiple DataLoader workers.
        d = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        self.u = d['tensor']        # (N, 201, 1024), float32
        self.nu = d['nu'].float().clone()   # (N,) - clone to avoid mmap deps
        assert self.u.shape[1] == 201 and self.u.shape[2] == 1024, \
            f"unexpected trajectory shape {self.u.shape}"

        self.B_t, self.B_x, self.M_x = B_t, B_x, M_x
        self.low_freqs = low_freqs
        self.T_pad = T_pad
        self.ic_repeat = ic_repeat
        self.gen_start = gen_start
        self.ic_block_rows = ic_block_rows
        self.u_scale = float(u_scale)
        self.Y_bound = float(Y_bound)
        self.train = train
        # Per-frequency std vector for whitening. Shape (feature_dim,).
        # When set, encode divides by this vector instead of the scalar Y_bound;
        # decode multiplies by it. The result: every coefficient has unit
        # variance, so unweighted MSE on the noise prediction is well-scaled.
        if per_freq_std is not None:
            self.coef_std = torch.as_tensor(per_freq_std, dtype=torch.float32)
        else:
            self.coef_std = None

        # Sanity: layout fits T_pad with no overlap.
        assert ic_repeat <= gen_start, "IC rows would overlap the gen rows"
        assert gen_start + 200 <= T_pad, "gen rows do not fit in T_pad"
        assert gen_start % B_t == 0, "gen_start must align to B_t"
        # ic_block_rows * B_t = clean rows at the front
        assert ic_block_rows * B_t == gen_start, \
            "ic_block_rows must cover all clean rows up to gen_start"

        self.X = self.u.shape[2]
        self.n_t = T_pad // B_t
        self.n_x = self.X // B_x
        assert self.n_x % M_x == 0
        self.n_x_macro = self.n_x // M_x
        self.n_tokens = self.n_t * self.n_x_macro
        self.feature_dim = M_x * low_freqs
        self.n_ic_tokens = ic_block_rows * self.n_x_macro

        self.zigzag = zigzag_order_2d(B_t, B_x)
        self.reverse_zigzag = reverse_zigzag_order_2d(B_t, B_x)
        self._dct_op = DCT2DBlocks(B_t, B_x)

        # Boolean mask over tokens; True for tokens in the leading
        # `ic_block_rows` time-blocks (these are the clean conditioning region).
        self._ic_mask = ic_token_mask(self.n_t, self.n_x_macro,
                                      ic_block_rows=ic_block_rows)

    def __len__(self):
        return self.u.shape[0]

    def encode_field(self, field):
        """field: (..., T_pad, X). Returns (..., n_tokens, feature_dim).

        If `coef_std` is set (per-coefficient whitening), each output channel
        is divided by its own std (so all channels become unit-variance).
        Otherwise we fall back to the scalar `Y_bound`.
        """
        field = field / self.u_scale
        toks = field_to_tokens(
            field, self.B_t, self.B_x, self.M_x, self.low_freqs,
            self.zigzag, self._dct_op,
        )
        if self.coef_std is not None:
            cs = self.coef_std.to(toks.device).clamp(min=1e-6)
            return toks / cs
        return toks / self.Y_bound

    def decode_tokens(self, toks):
        """Inverse of encode_field: tokens -> (..., T_pad, X) field in u units."""
        if self.coef_std is not None:
            cs = self.coef_std.to(toks.device).clamp(min=1e-6)
            scaled = toks * cs
        else:
            scaled = toks * self.Y_bound
        from DCT_utils_1d import tokens_to_field
        field = tokens_to_field(
            scaled, self.B_t, self.B_x, self.M_x, self.low_freqs,
            self.reverse_zigzag, n_t=self.n_t, n_x=self.n_x, dct_op=self._dct_op,
        )
        return field * self.u_scale

    def __getitem__(self, idx):
        u = self.u[idx]              # (201, 1024) — may be a view of the mmap
        u = u.clone()                # detach from mmap before further work
        nu = float(self.nu[idx])

        ic = u[0:1]                  # (1, 1024)
        gen = u[1:201]               # (200, 1024)

        # Layout: rows 0..ic_repeat-1 = IC repeated; rows ic_repeat..gen_start-1
        # = zeros (clean pad); rows gen_start..gen_start+200-1 = gen target.
        field = u.new_zeros(self.T_pad, self.X)
        field[: self.ic_repeat] = ic
        field[self.gen_start : self.gen_start + 200] = gen

        tokens = self.encode_field(field)                       # (n_tokens, feature_dim)
        return tokens, torch.tensor(nu, dtype=torch.float32), self._ic_mask

    def encode_ic(self, ic_batch):
        """Encode just the IC + zero-pad into the clean-token slice.
        ic_batch: (B, X). Returns (B, n_ic_tokens, feature_dim) — i.e. the
        tokens corresponding to the leading `ic_block_rows` time-blocks."""
        B = ic_batch.shape[0]
        full = ic_batch.new_zeros(B, self.T_pad, self.X)
        full[:, : self.ic_repeat] = ic_batch.unsqueeze(1).expand(
            B, self.ic_repeat, self.X)
        toks = self.encode_field(full)                          # (B, n_tokens, F)
        return toks[:, : self.n_ic_tokens]                      # (B, n_ic_tokens, F)


class Burgers1D:
    """DatasetFactory-style wrapper, mirrors DCTdiff/datasets.py conventions."""

    def __init__(
        self,
        path, test_path=None,
        B_t=8, B_x=8, M_x=4, low_freqs=8,
        T_pad=240, ic_repeat=8, gen_start=40, ic_block_rows=5,
        u_scale=0.6342716217041016, Y_bound=2.6,
        per_freq_std=None,
        **_,
    ):
        kw = dict(B_t=B_t, B_x=B_x, M_x=M_x, low_freqs=low_freqs,
                  T_pad=T_pad, ic_repeat=ic_repeat, gen_start=gen_start,
                  ic_block_rows=ic_block_rows,
                  u_scale=u_scale, Y_bound=Y_bound,
                  per_freq_std=per_freq_std)
        self.train_ds = Burgers1DDataset(path, train=True, **kw)
        self.test_ds = (
            Burgers1DDataset(test_path, train=False, **kw)
            if test_path and os.path.exists(test_path) else None
        )

    @property
    def data_shape(self):
        return self.train_ds.n_tokens, self.train_ds.feature_dim

    def get_split(self, split, labeled=False):
        return self.train_ds if split == 'train' else self.test_ds
