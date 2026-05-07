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


class Reaction1DDataset(Dataset):
    def __init__(
        self,
        path,
        B_t=16, B_x=32, M_x=4, low_freqs=32,
        T_pad=240, ic_repeat=8, gen_start=40, ic_block_rows=5,
        shift=0.5,                    # field is bounded in [0, 1]; center it
        u_scale=0.2541198134,         # std of (u - shift) over the dataset
        Y_bound=3.0,                  # ~ overall std of tokens after /u_scale
        train=True,
    ):
        super().__init__()
        d = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        self.u = d['tensor']                       # (N, 201, 1024), float32
        self.nu = d['nu'].float().clone()          # (N,)
        self.rho = d['rho'].float().clone()        # (N,)
        assert self.u.shape[1] == 201 and self.u.shape[2] == 1024, \
            f"unexpected trajectory shape {self.u.shape}"

        self.B_t, self.B_x, self.M_x = B_t, B_x, M_x
        self.low_freqs = low_freqs
        self.T_pad = T_pad
        self.ic_repeat = ic_repeat
        self.gen_start = gen_start
        self.ic_block_rows = ic_block_rows
        self.shift = float(shift)
        self.u_scale = float(u_scale)
        self.Y_bound = float(Y_bound)
        self.train = train

        assert ic_repeat <= gen_start, "IC rows would overlap the gen rows"
        assert gen_start + 200 <= T_pad, "gen rows do not fit in T_pad"
        assert gen_start % B_t == 0, "gen_start must align to B_t"
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

        self._ic_mask = ic_token_mask(self.n_t, self.n_x_macro,
                                      ic_block_rows=ic_block_rows)

    def __len__(self):
        return self.u.shape[0]

    def encode_field(self, field):
        """field: (..., T_pad, X), already shift-centered (mean ~0).
        Returns (..., n_tokens, feature_dim) normalized by Y_bound."""
        field = field / self.u_scale
        toks = field_to_tokens(
            field, self.B_t, self.B_x, self.M_x, self.low_freqs,
            self.zigzag, self._dct_op,
        )
        return toks / self.Y_bound

    def __getitem__(self, idx):
        u = self.u[idx]                  # (201, 1024) — may be a view of mmap
        u = u.clone()
        nu = float(self.nu[idx])
        rho = float(self.rho[idx])

        # Center around `shift` so 0 in centered space corresponds to the
        # natural midpoint of the bounded field. Pad rows are then plain zeros.
        u_c = u - self.shift
        ic_c = u_c[0:1]                  # (1, X)
        gen_c = u_c[1:201]               # (200, X)

        field = u.new_zeros(self.T_pad, self.X)
        field[: self.ic_repeat] = ic_c
        field[self.gen_start : self.gen_start + 200] = gen_c

        tokens = self.encode_field(field)
        return (
            tokens,
            torch.tensor(nu, dtype=torch.float32),
            torch.tensor(rho, dtype=torch.float32),
            self._ic_mask,
        )

    def encode_ic(self, ic_batch):
        """Encode just the IC + zero-pad into the clean-token slice.
        ic_batch: (B, X) in physical units (in [0, 1]).
        Returns (B, n_ic_tokens, feature_dim)."""
        B = ic_batch.shape[0]
        ic_c = ic_batch - self.shift
        full = ic_c.new_zeros(B, self.T_pad, self.X)
        full[:, : self.ic_repeat] = ic_c.unsqueeze(1).expand(
            B, self.ic_repeat, self.X)
        toks = self.encode_field(full)
        return toks[:, : self.n_ic_tokens]


class Reaction1D:
    """DatasetFactory-style wrapper, mirrors datasets_burgers.Burgers1D."""

    def __init__(
        self,
        path, test_path=None,
        B_t=8, B_x=8, M_x=4, low_freqs=8,
        T_pad=240, ic_repeat=8, gen_start=40, ic_block_rows=5,
        shift=0.5,
        u_scale=0.2541198134, Y_bound=3.0,
        **_,
    ):
        kw = dict(B_t=B_t, B_x=B_x, M_x=M_x, low_freqs=low_freqs,
                  T_pad=T_pad, ic_repeat=ic_repeat, gen_start=gen_start,
                  ic_block_rows=ic_block_rows,
                  shift=shift, u_scale=u_scale, Y_bound=Y_bound)
        self.train_ds = Reaction1DDataset(path, train=True, **kw)
        self.test_ds = (
            Reaction1DDataset(test_path, train=False, **kw)
            if test_path and os.path.exists(test_path) else None
        )

    @property
    def data_shape(self):
        return self.train_ds.n_tokens, self.train_ds.feature_dim

    def get_split(self, split, labeled=False):
        return self.train_ds if split == 'train' else self.test_ds
