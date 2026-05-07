
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from DCT_utils import (
    split_into_blocks, zigzag_order,
    split_clip_into_blocks_3d, dct3_block, zigzag_order_3d,
)


class DatasetFactory:
    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        dataset = self.train if split == "train" else self.test
        assert not labeled, "MovingMNIST is unlabeled"
        return dataset

    @property
    def has_label(self):
        return False

    @property
    def fid_stat(self):
        return None

    @property
    def data_shape(self):
        raise NotImplementedError


def _encode_frame(frame: np.ndarray, block_sz: int, num_blocks: int,
                  low2high_order, low_freqs: int, y_bound: float) -> np.ndarray:
    """frame: (H, W) float32 in [0, 1] → (num_blocks, low_freqs) float32."""
    blocks = split_into_blocks(frame, block_sz).astype(np.float32, copy=False)
    dct_blocks = np.empty_like(blocks)
    for i, blk in enumerate(blocks):
        dct_blocks[i] = cv2.dct(blk)
    dct_blocks = dct_blocks.reshape(num_blocks, block_sz * block_sz)
    dct_blocks = dct_blocks[:, low2high_order][:, :low_freqs]
    return dct_blocks / y_bound


class MovingMNISTFrameDataset(Dataset):
    """Loads a pre-generated Moving MNIST .pt file.

    The .pt file is expected to be a tensor of shape (N, T, H, W) with
    pixel values in [0, 255].  Each __getitem__ returns one DCT-tokenized
    frame; indices map flatly across all sequences and timesteps.
    """

    def __init__(self, data_path: str, tokens: int, low_freqs: int,
                 block_sz: int, y_bound: float, train_frac: float = 0.9):
        data = torch.load(data_path, map_location='cpu', weights_only=False)

        # Support both (N, T, H, W) and (N, T, 1, H, W)
        if data.dim() == 5:
            data = data.squeeze(2)
        assert data.dim() == 4, f"expected (N, T, H, W), got {tuple(data.shape)}"

        N, T, H, W = data.shape
        assert H == W, f"expected square frames, got {H}x{W}"

        # Train/val split on the sequence (not frame) axis
        n_train = int(N * train_frac)
        self._data = data         # (N, T, H, W)
        self._n_train = n_train
        self._N = N
        self._T = T

        resolution = H
        self.tokens = tokens
        self.low_freqs = low_freqs
        self.block_sz = block_sz
        self.y_bound = float(y_bound)
        self.low2high_order = zigzag_order(block_sz)
        self.num_blocks = (resolution // block_sz) ** 2
        assert self.num_blocks == tokens, (
            f"tokens={tokens} but (resolution/block_sz)^2 = {self.num_blocks}; "
            f"check resolution={resolution} and block_sz={block_sz}"
        )
        self._train = True   # toggled by _as_split()

    def _as_split(self, train: bool):
        obj = object.__new__(MovingMNISTFrameDataset)
        obj.__dict__.update(self.__dict__)
        obj._train = train
        return obj

    def __len__(self):
        n_seqs = self._n_train if self._train else (self._N - self._n_train)
        return n_seqs * self._T

    def __getitem__(self, idx):
        n_seqs = self._n_train if self._train else (self._N - self._n_train)
        seq_local = idx // self._T
        t         = idx  % self._T
        seq_idx   = seq_local if self._train else (self._n_train + seq_local)
        frame = self._data[seq_idx, t].numpy().astype(np.float32)
        tokens = _encode_frame(frame, self.block_sz, self.num_blocks,
                                self.low2high_order, self.low_freqs, self.y_bound)
        return torch.from_numpy(tokens)   # (tokens, low_freqs)


class MovingMNIST(DatasetFactory):
    """DCTdiff DatasetFactory for Moving MNIST (single-channel, unlabeled).

    Args:
        data_path: path to the .pt file, e.g.
            /scratch/bgxp/factor_diffusion/original_data/moving_mnist/moving_mnist_20k_2slow.pt
        resolution: frame size in pixels (must equal H=W in the file)
        tokens: (resolution // block_sz) ** 2
        low_freqs: number of zigzag DCT coefficients to keep per block
        block_sz: spatial block size for 2D DCT (e.g. 8)
        Y_bound: scalar used to normalise DCT coefficients into ~[-1, 1]
        train_frac: fraction of sequences used for training (rest = val)
    """

    def __init__(self, data_path: str, resolution: int = 64, tokens: int = 0,
                 low_freqs: int = 0, block_sz: int = 8, Y_bound=None,
                 train_frac: float = 0.9, channels: int = 1, **kwargs):
        super().__init__()
        if channels != 1:
            raise ValueError(f"MovingMNIST expects channels=1, got {channels}")
        if Y_bound is None:
            raise ValueError("MovingMNIST requires Y_bound")

        self._tokens = tokens
        self._low_freqs = low_freqs
        self.channels = channels

        y_bound = float(np.asarray(Y_bound, dtype=np.float32).reshape(-1)[0])
        base = MovingMNISTFrameDataset(
            data_path=data_path, tokens=tokens, low_freqs=low_freqs,
            block_sz=block_sz, y_bound=y_bound, train_frac=train_frac,
        )
        self.train = base._as_split(train=True)
        self.test  = base._as_split(train=False)

    @property
    def data_shape(self):
        return self._tokens, self._low_freqs * self.channels


def _encode_clip_3d(clip: np.ndarray, block_T: int, block_HW: int,
                    zz_order, low_freqs: int, y_bound: float) -> np.ndarray:
    """clip: (T, H, W) float32 → (num_blocks, low_freqs) float32, normalized."""
    blocks = split_clip_into_blocks_3d(clip, block_T, block_HW, block_HW)
    n = blocks.shape[0]
    coefs = np.empty((n, low_freqs), dtype=np.float32)
    flat = block_T * block_HW * block_HW
    for i, blk in enumerate(blocks):
        coefs[i] = dct3_block(blk).reshape(flat)[zz_order][:low_freqs]
    return coefs / y_bound


class MovingMNISTClipDataset(Dataset):
    """Loads Moving MNIST .pt and yields one 3D-DCT-tokenized clip per sample.

    Output shape per sample: (num_blocks_per_clip, low_freqs) where
        num_blocks_per_clip = (T/block_T) * (H/block_HW)^2
    """

    def __init__(self, data_path: str, block_T: int, block_HW: int,
                 low_freqs: int, y_bound: float, train_frac: float = 1.0):
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        if data.dim() == 5:
            data = data.squeeze(2)
        assert data.dim() == 4, f"expected 4D tensor, got {tuple(data.shape)}"
        # Normalize axis order to (N, T, H, W). The .pt is stored as (T, N, H, W)
        # for Moving MNIST, so transpose if the first axis is smaller than the second.
        if data.shape[0] < data.shape[1]:
            data = data.permute(1, 0, 2, 3).contiguous()
        N, T, H, W = data.shape
        assert H == W
        assert T % block_T == 0 and H % block_HW == 0, (
            f"clip ({T},{H},{W}) not divisible by ({block_T},{block_HW},{block_HW})"
        )

        self._data = data
        self._N, self._T, self._H = N, T, H
        self._n_train = int(N * train_frac) if train_frac < 1.0 else N
        self._train = True

        self.block_T = block_T
        self.block_HW = block_HW
        self.low_freqs = low_freqs
        self.y_bound = float(y_bound)
        self.zz_order = zigzag_order_3d(block_T, block_HW, block_HW)
        self.num_blocks_per_clip = (T // block_T) * (H // block_HW) * (W // block_HW)

    def _as_split(self, train: bool):
        obj = object.__new__(MovingMNISTClipDataset)
        obj.__dict__.update(self.__dict__)
        obj._train = train
        return obj

    def __len__(self):
        if self._train:
            return self._n_train
        return self._N - self._n_train

    def __getitem__(self, idx):
        seq_idx = idx if self._train else (self._n_train + idx)
        clip = self._data[seq_idx].numpy().astype(np.float32)  # (T, H, W)
        tokens = _encode_clip_3d(clip, self.block_T, self.block_HW,
                                 self.zz_order, self.low_freqs, self.y_bound)
        return torch.from_numpy(tokens)


class MovingMNIST3D(DatasetFactory):
    """DCTdiff DatasetFactory for 3D-DCT-tokenized Moving MNIST clips."""

    def __init__(self, data_path: str, block_T: int = 4, block_HW: int = 8,
                 low_freqs: int = 40, Y_bound=None, train_frac: float = 1.0,
                 channels: int = 1, **kwargs):
        super().__init__()
        if channels != 1:
            raise ValueError(f"MovingMNIST3D expects channels=1, got {channels}")
        if Y_bound is None:
            raise ValueError("MovingMNIST3D requires Y_bound")

        self._low_freqs = low_freqs
        self.channels = channels

        y_bound = float(np.asarray(Y_bound, dtype=np.float32).reshape(-1)[0])
        base = MovingMNISTClipDataset(
            data_path=data_path, block_T=block_T, block_HW=block_HW,
            low_freqs=low_freqs, y_bound=y_bound, train_frac=train_frac,
        )
        self._tokens = base.num_blocks_per_clip
        self.train = base._as_split(train=True)
        self.test  = base._as_split(train=False)

    @property
    def data_shape(self):
        return self._tokens, self._low_freqs * self.channels
