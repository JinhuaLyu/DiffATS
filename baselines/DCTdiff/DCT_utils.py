import numpy as np
import cv2
from scipy.fft import dctn, idctn


def split_into_blocks(image, block_sz):
    blocks = []
    for i in range(0, image.shape[0], block_sz):
        for j in range(0, image.shape[1], block_sz):
            blocks.append(image[i:i + block_sz, j:j + block_sz])  # first row, then column
    return np.array(blocks)

def combine_blocks(blocks, height, width, block_sz):
    image = np.zeros((height, width), np.float32)
    index = 0
    for i in range(0, height, block_sz):
        for j in range(0, width, block_sz):
            image[i:i + block_sz, j:j + block_sz] = blocks[index]
            index += 1
    return image

def dct_transform(blocks):
    dct_blocks = []
    for block in blocks:
        dct_block = np.float32(block) - 128  # Shift to center around 0
        dct_block = cv2.dct(dct_block)
        dct_blocks.append(dct_block)
    return np.array(dct_blocks)

def idct_transform(blocks):
    idct_blocks = []
    for block in blocks:
        idct_block = cv2.idct(block)
        idct_block = idct_block + 128  # Shift back
        idct_blocks.append(idct_block)
    return np.array(idct_blocks)


def zigzag_order(block_sz=8):
    index_list = []

    # Iterate over each diagonal defined by the sum of row and column indices
    for s in range(2 * (block_sz - 1) + 1):
        temp = []  # Initialize a temporary list to collect indices in the current diagonal
        start = max(0, s - block_sz + 1)  # Calculate starting and ending points of the diagonal
        end = min(s, block_sz - 1)

        for i in range(start, end + 1):  # Collect indices in the current diagonal
            temp.append((i, s - i))

        if s % 2 == 0:  # Reverse the diagonal elements if the sum of indices is even
            temp.reverse()

        index_list.extend(temp)  # Convert 2D indices to 1D and append to the main list

    return [i * block_sz + j for i, j in index_list]  # Convert tuple (i, j) to index i * B + j


def reverse_zigzag_order(block_sz=8):
    zigzag_indices = zigzag_order(block_sz)  # Get the zigzag order list
    reverse_order = [0] * (block_sz * block_sz)  # Initialize an array of the same size to store the reverse order

    # Populate the reverse order list where the index is the original position,
    # and the value is the new position according to the zigzag order
    for index, value in enumerate(zigzag_indices):
        reverse_order[value] = index

    return reverse_order




# ---------------------------------------------------------------------------
# 3D DCT primitives (used for (T, H, W) volumetric clips, e.g. Burgers2D, Karman, MovingMNIST)
# ---------------------------------------------------------------------------

def split_clip_into_blocks_3d(clip, block_T, block_H, block_W):
    T, H, W = clip.shape
    assert T % block_T == 0 and H % block_H == 0 and W % block_W == 0, (
        f"clip {clip.shape} not divisible by block ({block_T},{block_H},{block_W})"
    )
    nT, nH, nW = T // block_T, H // block_H, W // block_W
    blocks = clip.reshape(nT, block_T, nH, block_H, nW, block_W)
    blocks = blocks.transpose(0, 2, 4, 1, 3, 5)
    return blocks.reshape(nT * nH * nW, block_T, block_H, block_W)


def combine_blocks_3d(blocks, T, H, W, block_T, block_H, block_W):
    nT, nH, nW = T // block_T, H // block_H, W // block_W
    assert blocks.shape[0] == nT * nH * nW
    blocks = blocks.reshape(nT, nH, nW, block_T, block_H, block_W)
    blocks = blocks.transpose(0, 3, 1, 4, 2, 5)
    return blocks.reshape(T, H, W)


def dct3_block(block):
    return dctn(block.astype(np.float32, copy=False), type=2, norm='ortho')


def idct3_block(block):
    return idctn(block.astype(np.float32, copy=False), type=2, norm='ortho')


def zigzag_order_3d(block_T, block_H, block_W):
    coords = []
    for i in range(block_T):
        for j in range(block_H):
            for k in range(block_W):
                coords.append((i + j + k, i, j, k))
    coords.sort()
    return [i * block_H * block_W + j * block_W + k for _, i, j, k in coords]


def reverse_zigzag_order_3d(block_T, block_H, block_W):
    fwd = zigzag_order_3d(block_T, block_H, block_W)
    rev = [0] * len(fwd)
    for new_pos, old_pos in enumerate(fwd):
        rev[old_pos] = new_pos
    return rev


# ---------------------------------------------------------------------------
# 2D zigzag order (for 2D image patches; used in eval/sampling)
# ---------------------------------------------------------------------------

def zigzag_order_2d(block_H, block_W=None):
    """Return zigzag scan indices for a (block_H, block_W) patch.
    When block_W is None, falls back to the square (block_H x block_H) version."""
    if block_W is None:
        return zigzag_order(block_H)
    index_list = []
    for s in range(block_H + block_W - 1):
        temp = []
        start_h = max(0, s - block_W + 1)
        end_h = min(s, block_H - 1)
        for i in range(start_h, end_h + 1):
            temp.append((i, s - i))
        if s % 2 == 0:
            temp.reverse()
        index_list.extend(temp)
    return [i * block_W + j for i, j in index_list]