from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import math
import random
from PIL import Image
import os
import glob
import einops
import torchvision.transforms.functional as F
import cv2
from DCT_utils import (
    split_into_blocks, dct_transform, combine_blocks, idct_transform,
    zigzag_order, reverse_zigzag_order,
    split_clip_into_blocks_3d, dct3_block, zigzag_order_3d,
)


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]


class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if random.random() < self.p_uncond:
            y = self.empty_token
        return x, y


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


# CIFAR10

class CIFAR10(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)])
        self.train = DCT_4YCbCr(
            root_dir=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6  # (96, 43)

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return '/data/scratch/U-ViT2/assets/fid_stats/fid_stats_cifar10_train.npz'

    @property
    def has_label(self):
        return False



# ImageNet


class FeatureDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        # names = sorted(os.listdir(path))
        # self.files = [os.path.join(path, name) for name in names]

    def __len__(self):
        return 1_281_167 * 2  # consider the random flip

    def __getitem__(self, idx):
        path = os.path.join(self.path, f'{idx}.npy')
        z, label = np.load(path, allow_pickle=True)
        return z, label


class ImageNet256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet512Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 64, 64

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet512_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet(DatasetFactory):
    def __init__(self, path, resolution, random_crop=False, random_flip=True):
        super().__init__()

        print(f'Counting ImageNet files from {path}')
        train_files = _list_image_files_recursively(os.path.join(path, 'train'))
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting ImageNet files')

        self.train = ImageDataset(resolution, train_files, labels=train_labels, random_crop=random_crop, random_flip=random_flip)
        self.resolution = resolution
        if len(self.train) != 1_281_167:
            print(f'Missing train samples: {len(self.train)} < 1281167')

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet{self.resolution}_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.listdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        labels,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths
        self.labels = labels
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        label = np.array(self.labels[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), label


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


# CelebA


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class DCT_4YCbCr(Dataset):
    def __init__(self, root_dir, img_sz=64, tokens=0, low_freqs=0, block_sz=8, Y_bound=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.img_paths = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.img_paths.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

        # parameters of DCT design
        self.Y_bound = np.array(Y_bound)
        print(f"using eta {self.Y_bound} for training")
        self.tokens = tokens
        self.low_freqs = low_freqs
        self.block_sz = block_sz

        Y = int(img_sz * img_sz / (block_sz * block_sz))  # num of Y blocks
        self.Y_blocks_per_row = int(img_sz / block_sz)
        self.index = []  # index of Y if merging 2*2 Y-block area
        for row in range(0, Y, int(2 * self.Y_blocks_per_row)):  # 0, 32, 64...
            for col in range(0, self.Y_blocks_per_row, 2):  # 0, 2, 4...
                self.index.append(row + col)
        assert len(self.index) == int(Y / 4)

        self.low2high_order = zigzag_order(block_sz)
        self.reverse_order = reverse_zigzag_order(block_sz)

        # token sequence: 4Y-Cb-Cr-4Y-Cb-Cr...
        self.cb_index = [i for i in range(4, tokens, 6)]
        self.cr_index = [i for i in range(5, tokens, 6)]
        self.y_index = [i for i in range(0, tokens) if i not in self.cb_index and i not in self.cr_index]
        assert len(self.y_index) + len(self.cb_index) + len(self.cr_index) == tokens

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # img.save('original_img.jpg')
        img = transforms.RandomHorizontalFlip()(img)  # do data augmentation by PIL
        img = np.array(img)

        # Step 1: Convert RGB to YCbCr
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        img_y = 0.299 * R + 0.587 * G + 0.114 * B
        img_cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
        img_cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

        cb_downsampled = cv2.resize(img_cb, (img_cb.shape[1] // 2, img_cb.shape[0] // 2),
                                    interpolation=cv2.INTER_LINEAR)
        cr_downsampled = cv2.resize(img_cr, (img_cr.shape[1] // 2, img_cr.shape[0] // 2),
                                    interpolation=cv2.INTER_LINEAR)

        # Step 2: Split the Y, Cb, and Cr components into BxB blocks
        y_blocks = split_into_blocks(img_y, self.block_sz)  # Y component, (h, w) --> (h/B * w/B, B, B)
        cb_blocks = split_into_blocks(cb_downsampled, self.block_sz)  # Cb component, (h/2, w/2) --> (h/2B * w/2B, B, B)
        cr_blocks = split_into_blocks(cr_downsampled, self.block_sz)  # Cr component, (h/2, w/2) --> (h/2B * w/2B, B, B)

        # Step 3: Apply DCT on each block
        dct_y_blocks = dct_transform(y_blocks)  # (h/B * w/B, B, B)
        dct_cb_blocks = dct_transform(cb_blocks)  # (h/2B * w/2B, B, B)
        dct_cr_blocks = dct_transform(cr_blocks)  # (h/2B * w/2B, B, B)

        # Step 4: organize the token order by Y-Y-Y-Y-Cb-Cr (2_blocks*2_blocks region)
        DCT_blocks = []
        for i in range(dct_cr_blocks.shape[0]):
            DCT_blocks.append([
                dct_y_blocks[self.index[i]],  # Y
                dct_y_blocks[self.index[i] + 1],  # Y
                dct_y_blocks[self.index[i] + self.Y_blocks_per_row],  # Y
                dct_y_blocks[self.index[i] + self.Y_blocks_per_row + 1],  # Y
                dct_cb_blocks[i],  # Cb
                dct_cr_blocks[i],  # Cr
            ])
        DCT_blocks = np.array(DCT_blocks).reshape(-1, 6, self.block_sz*self.block_sz)  # (num_tokens, 6, B**2)

        # Step 5: scale into [-1, 1]
        assert DCT_blocks.shape == (self.tokens, 6, self.block_sz*self.block_sz)
        DCT_blocks[:, :4 :] = (DCT_blocks[:, :4 :]) / self.Y_bound
        DCT_blocks[:, 4, :] = (DCT_blocks[:, 4, :]) / self.Y_bound
        DCT_blocks[:, 5, :] = (DCT_blocks[:, 5, :]) / self.Y_bound

        # Step 6: reorder coe from low to high freq, then mask out high-freq signals
        DCT_blocks = DCT_blocks[:, :, self.low2high_order]  # (num_tokens, 6, B**2)
        DCT_blocks = DCT_blocks[:, :, :self.low_freqs]  # (num_tokens, 6, B**2) --> (num_tokens, 6, low_freq_coe)

        # numpy to torch
        DCT_blocks = torch.from_numpy(DCT_blocks).reshape(self.tokens, -1)  # (num_tokens, 6*low_freq_coe)
        DCT_blocks = DCT_blocks.float()  # float64 --> float32

        return DCT_blocks

        # """debug DCT to RGB"""
        # tokens = self.tokens
        # low_freqs = self.low_freqs
        # block_sz = self.block_sz
        # reverse_order = self.reverse_order
        # resolution=64
        # cb_blocks_per_row = int((resolution / block_sz) / 2)
        # Y_blocks_per_row = int(resolution / block_sz)
        #
        # cb_index = [i for i in range(4, tokens, 6)]
        # cr_index = [i for i in range(5, tokens, 6)]
        # y_index = [i for i in range(0, tokens) if i not in cb_index and i not in cr_index]
        # assert len(y_index) + len(cb_index) + len(cr_index) == tokens
        # y_tokens = int((tokens / 6) * 4)
        # cb_tokens = int(tokens / 6)
        #
        # sample = DCT_blocks.numpy()
        # assert sample.shape == (tokens, low_freqs)
        # # sample = np.clip(sample, -2, 2)  # clamp into [-1, 1]
        #
        # # fill up DCT coes
        # DCT = np.zeros((tokens, block_sz * block_sz))
        # DCT[:, :low_freqs] = sample
        # DCT = DCT[:, reverse_order]  # convert the low to high freq order back to sequential order
        #
        # # DCT_Cb = ((DCT[cb_index, :] * CELEBA64_Cb_99_centered) + CELEBA64_Cb_mean)
        # # DCT_Cr = ((DCT[cr_index, :] * CELEBA64_Cr_99_centered) + CELEBA64_Cr_mean)
        # DCT_Cb = DCT[cb_index, :]
        # DCT_Cr = DCT[cr_index, :]
        # DCT_Cb = DCT_Cb.reshape(cb_tokens, block_sz, block_sz)  # (16, 64) --> (16, 8, 8)
        # DCT_Cr = DCT_Cr.reshape(cb_tokens, block_sz, block_sz)  # (16, 64) --> (16, 8, 8)
        #
        # y_blocks = []
        # # DCT_Y = (DCT[y_index, :] * CELEBA64_Y_99_centered) + CELEBA64_Y_mean  # (64, 64)
        # DCT_Y = DCT[y_index, :]
        # for row in range(cb_blocks_per_row):  # 16 cb/cr blocks, so 4*4 spatial blocks
        #     tem_ls = []
        #     for col in range(cb_blocks_per_row):
        #         ind = row * (Y_blocks_per_row * 2) + col * 4
        #         y_blocks.append(DCT_Y[ind, :])
        #         y_blocks.append(DCT_Y[ind + 1, :])
        #         tem_ls.append(DCT_Y[ind + 2, :])
        #         tem_ls.append(DCT_Y[ind + 3, :])
        #     for ele in tem_ls:
        #         y_blocks.append(ele)
        # DCT_Y = np.array(y_blocks).reshape(y_tokens, block_sz, block_sz)
        #
        # # Apply Inverse DCT on each block
        # idct_y_blocks = idct_transform(DCT_Y)
        # idct_cb_blocks = idct_transform(DCT_Cb)
        # idct_cr_blocks = idct_transform(DCT_Cr)
        #
        # # Combine blocks back into images
        # height, width = resolution, resolution
        # y_reconstructed = combine_blocks(idct_y_blocks, height, width, block_sz)
        # cb_reconstructed = combine_blocks(idct_cb_blocks, int(height / 2), int(width / 2), block_sz)
        # cr_reconstructed = combine_blocks(idct_cr_blocks, int(height / 2), int(width / 2), block_sz)
        #
        # # Upsample Cb and Cr to original size
        # cb_upsampled = cv2.resize(cb_reconstructed, (width, height), interpolation=cv2.INTER_LINEAR)
        # cr_upsampled = cv2.resize(cr_reconstructed, (width, height), interpolation=cv2.INTER_LINEAR)
        #
        # # Step 5: Convert YCbCr back to RGB
        # R = y_reconstructed + 1.402 * (cr_upsampled - 128)
        # G = y_reconstructed - 0.344136 * (cb_upsampled - 128) - 0.714136 * (cr_upsampled - 128)
        # B = y_reconstructed + 1.772 * (cb_upsampled - 128)
        #
        # rgb_reconstructed = np.zeros((height, width, 3))
        # rgb_reconstructed[:, :, 0] = np.clip(R, 0, 255)
        # rgb_reconstructed[:, :, 1] = np.clip(G, 0, 255)
        # rgb_reconstructed[:, :, 2] = np.clip(B, 0, 255)
        #
        # # Convert to uint8
        # rgb_reconstructed = np.uint8(rgb_reconstructed)  # (h, w, 3), RGB channels
        # final_image = Image.fromarray(rgb_reconstructed)
        # final_image.save('recon_pure_manual.jpg')
        # time.sleep(3)
        # raise ValueError


class DCT_4YCbCr_cond(Dataset):
    def __init__(self, img_sz=64, tokens=0, low_freqs=0, block_sz=8, train_files=None, labels=None, Y_bound=None):

        self.image_paths = train_files
        self.labels = labels

        # parameters of DCT design
        self.Y_bound = np.array(Y_bound)
        print(f"using eta {self.Y_bound} for training")
        self.tokens = tokens
        self.low_freqs = low_freqs
        self.block_sz = block_sz

        Y = int(img_sz * img_sz / (block_sz * block_sz))  # num of Y blocks
        self.Y_blocks_per_row = int(img_sz / block_sz)
        self.index = []  # index of Y if merging 2*2 Y-block area
        for row in range(0, Y, int(2 * self.Y_blocks_per_row)):  # 0, 32, 64...
            for col in range(0, self.Y_blocks_per_row, 2):  # 0, 2, 4...
                self.index.append(row + col)
        assert len(self.index) == int(Y / 4)

        self.low2high_order = zigzag_order(block_sz)
        self.reverse_order = reverse_zigzag_order(block_sz)

        # token sequence: 4Y-Cb-Cr-4Y-Cb-Cr...
        self.cb_index = [i for i in range(4, tokens, 6)]
        self.cr_index = [i for i in range(5, tokens, 6)]
        self.y_index = [i for i in range(0, tokens) if i not in self.cb_index and i not in self.cr_index]
        assert len(self.y_index) + len(self.cb_index) + len(self.cr_index) == tokens

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # img.save('original_img.jpg')
        img = transforms.RandomHorizontalFlip()(img)  # do data augmentation by PIL
        img = np.array(img)

        # Step 1: Convert RGB to YCbCr
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        img_y = 0.299 * R + 0.587 * G + 0.114 * B
        img_cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
        img_cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

        cb_downsampled = cv2.resize(img_cb, (img_cb.shape[1] // 2, img_cb.shape[0] // 2),
                                    interpolation=cv2.INTER_LINEAR)
        cr_downsampled = cv2.resize(img_cr, (img_cr.shape[1] // 2, img_cr.shape[0] // 2),
                                    interpolation=cv2.INTER_LINEAR)

        # Step 2: Split the Y, Cb, and Cr components into BxB blocks
        y_blocks = split_into_blocks(img_y, self.block_sz)  # Y component, (h, w) --> (h/B * w/B, B, B)
        cb_blocks = split_into_blocks(cb_downsampled, self.block_sz)  # Cb component, (h/2, w/2) --> (h/2B * w/2B, B, B)
        cr_blocks = split_into_blocks(cr_downsampled, self.block_sz)  # Cr component, (h/2, w/2) --> (h/2B * w/2B, B, B)

        # Step 3: Apply DCT on each block
        dct_y_blocks = dct_transform(y_blocks)  # (h/B * w/B, B, B)
        dct_cb_blocks = dct_transform(cb_blocks)  # (h/2B * w/2B, B, B)
        dct_cr_blocks = dct_transform(cr_blocks)  # (h/2B * w/2B, B, B)

        # Step 4: organize the token order by Y-Y-Y-Y-Cb-Cr (2_blocks*2_blocks pixel region)
        DCT_blocks = []
        for i in range(dct_cr_blocks.shape[0]):
            DCT_blocks.append([
                dct_y_blocks[self.index[i]],  # Y
                dct_y_blocks[self.index[i] + 1],  # Y
                dct_y_blocks[self.index[i] + self.Y_blocks_per_row],  # Y
                dct_y_blocks[self.index[i] + self.Y_blocks_per_row + 1],  # Y
                dct_cb_blocks[i],  # Cb
                dct_cr_blocks[i],  # Cr
            ])
        DCT_blocks = np.array(DCT_blocks).reshape(-1, 6, self.block_sz * self.block_sz)  # (num_tokens, 6, B**2)

        # Step 5: scale into [-1, 1]
        assert DCT_blocks.shape == (self.tokens, 6, self.block_sz * self.block_sz)
        DCT_blocks[:, :4:] = DCT_blocks[:, :4:] / self.Y_bound
        DCT_blocks[:, 4, :] = DCT_blocks[:, 4, :] / self.Y_bound
        DCT_blocks[:, 5, :] = DCT_blocks[:, 5, :] / self.Y_bound

        # Step 6: reorder coe from low to high freq, then mask out high-freq signals
        DCT_blocks = DCT_blocks[:, :, self.low2high_order]  # organize freqs in the zigzag order
        DCT_blocks = DCT_blocks[:, :, :self.low_freqs]  # (num_tokens, 6, B**2) --> (num_tokens, 6, low_freq_coe)

        # numpy to torch
        DCT_blocks = torch.from_numpy(DCT_blocks).reshape(self.tokens, -1)  # (num_tokens, 6*low_freq_coe)
        DCT_blocks = DCT_blocks.float()  # float64 --> float32

        label = np.array(self.labels[idx], dtype=np.int64)

        return DCT_blocks, label


class CelebA(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        """
        manually download dataset: https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM&authuser=0
        then do center crop to 64x64 and set the image folder as the following 'path'
        """
        self.train = DCT_4YCbCr(
            root_dir=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return '/data/scratch/U-ViT2/assets/fid_stats/fid_stats_celeba64_all.npz'

    @property
    def has_label(self):
        return False


class FFHQ128(DatasetFactory):
    def __init__(self, path, resolution=128, tokens=0, low_freqs=0, block_sz=0, Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        self.train = DCT_4YCbCr(
            root_dir=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return '/data/scratch/U-ViT2/assets/fid_stats/fid_stats_ffhq128_jpg.npz'

    @property
    def has_label(self):
        return False


class FFHQ256(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        self.train = DCT_4YCbCr(
            root_dir=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return '/data/scratch/U-ViT2/assets/fid_stats/fid_stats_ffhq256_jpg.npz'

    @property
    def has_label(self):
        return False


class FFHQ512(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        self.train = DCT_4YCbCr(
            root_dir=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return '/data/scratch/U-ViT2/assets/fid_stats/fid_stats_ffhq512_jpg.npz'

    @property
    def has_label(self):
        return False


class AFHQ512(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        self.train = DCT_4YCbCr(
            root_dir=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return '/data/scratch/U-ViT2/assets/fid_stats/fid_stats_afhq512_jpg.npz'

    @property
    def has_label(self):
        return False


class ImageNet64(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        print(f'Counting ImageNet files from {path}')
        train_files = _list_image_files_recursively(path)
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting ImageNet files')

        self.train = DCT_4YCbCr_cond(
            img_sz=resolution, tokens=tokens, train_files=train_files, labels=train_labels,
            low_freqs=low_freqs, block_sz=block_sz, Y_bound=Y_bound,
        )

        if len(self.train) != 1_281_167:
            print(f'Missing train samples: {len(self.train)} < 1281167')

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6  # (96, 43)

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return f'/data/scratch/U-ViT2/assets/fid_stats/fid_stats_imgnet64_jpg.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


# MS COCO
def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


class MSCOCODatabase(Dataset):
    def __init__(self, root, annFile, size=None):
        from pycocotools.coco import COCO
        self.root = root
        self.height = self.width = size

        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        image = center_crop(self.width, self.height, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        anns = self._load_target(key)
        target = []
        for ann in anns:
            target.append(ann['caption'])

        return image, target


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


class MSCOCOFeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root):
        self.root = root
        self.num_data, self.n_captions = get_feature_dir_info(root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        z = np.load(os.path.join(self.root, f'{index}.npy'))
        k = random.randint(0, self.n_captions[index] - 1)
        c = np.load(os.path.join(self.root, f'{index}_{k}.npy'))
        return z, c


class MSCOCO256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = MSCOCOFeatureDataset(os.path.join(path, 'train'))
        self.test = MSCOCOFeatureDataset(os.path.join(path, 'val'))
        assert len(self.train) == 82783
        assert len(self.test) == 40504
        print('Prepare dataset ok')

        self.empty_context = np.load(os.path.join(path, 'empty_context.npy'))

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.empty_context)

        # text embedding extracted by clip
        # for visulization in t2i
        self.prompts, self.contexts = [], []
        for f in sorted(os.listdir(os.path.join(path, 'run_vis')), key=lambda x: int(x.split('.')[0])):
            prompt, context = np.load(os.path.join(path, 'run_vis', f), allow_pickle=True)
            self.prompts.append(prompt)
            self.contexts.append(context)
        self.contexts = np.array(self.contexts)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_mscoco256_val.npz'


def get_dataset(name, **kwargs):
    if name == 'cifar10':
        return CIFAR10(**kwargs)
    elif name == 'imagenet':
        return ImageNet(**kwargs)
    elif name == 'imagenet256_features':
        return ImageNet256Features(**kwargs)
    elif name == 'imagenet512_features':
        return ImageNet512Features(**kwargs)
    elif name == 'celeba':
        return CelebA(**kwargs)
    elif name == 'ffhq128':
        return FFHQ128(**kwargs)
    elif name == 'ffhq256':
        return FFHQ256(**kwargs)
    elif name == 'ffhq512':
        return FFHQ512(**kwargs)
    elif name == 'afhq512':
        return AFHQ512(**kwargs)
    elif name == 'imgnet64':
        return ImageNet64(**kwargs)
    elif name == 'mscoco256_features':
        return MSCOCO256Features(**kwargs)
    elif name == 'karman_vortex_3d':
        return KarmanVortex3D(**kwargs)
    elif name == 'karman_vortex_3d_cached':
        return KarmanVortex3DCached(**kwargs)
    elif name == 'burgers_2d':
        return BurgersVortex3D(**kwargs)
    elif name == 'burgers_2d_cached':
        return BurgersVortex3DCached(**kwargs)
    elif name == 'burgers_2d_t0cond_cached':
        return BurgersVortex3DT0CondCached(**kwargs)
    else:
        raise NotImplementedError(name)


# ---------------------------------------------------------------------------
# Karman vortex (2D simulation, T frames per clip) — 3D-DCT tokenization
# ---------------------------------------------------------------------------

def _list_pt_shards(root, prefix):
    return sorted(p for p in glob.glob(os.path.join(root, f'{prefix}*.pt')))


def _encode_clip_3d(clip, block_T, block_H, block_W, zz_order, low_freqs, y_bound):
    """clip: (T, H, W) float32 -> (num_blocks, low_freqs) float32, normalised.
    Uses one batched 3D DCT call across all blocks (~80x faster than a per-block loop)."""
    from scipy.fft import dctn
    blocks = split_clip_into_blocks_3d(clip, block_T, block_H, block_W)
    flat = block_T * block_H * block_W
    dct_blocks = dctn(blocks, type=2, norm='ortho', axes=(1, 2, 3))
    coefs = dct_blocks.reshape(blocks.shape[0], flat)[:, zz_order][:, :low_freqs]
    return (coefs / y_bound).astype(np.float32, copy=False)


class KarmanShardedClipDataset(Dataset):
    """Reads sharded .pt files; each shard is a list of dicts with key 'vor'
    of shape (T, H, W). Clips are 3D-DCT-tokenised on demand.

    Shards are assumed to hold a fixed number of clips (`clips_per_shard`,
    default 50). The first shard is opened on init to verify the count.
    """

    def __init__(self, shard_paths, T, block_T, block_H, block_W,
                 low_freqs, y_bound, drop_last_frame=True, clips_per_shard=50,
                 cache_in_memory=True):
        if not shard_paths:
            raise FileNotFoundError("no shards found")
        self._shards = list(shard_paths)
        self._cached_idx = -1
        self._cached_shard = None
        self.T = int(T)
        self.block_T = int(block_T)
        self.block_H = int(block_H)
        self.block_W = int(block_W)
        self.low_freqs = int(low_freqs)
        self.y_bound = float(y_bound)
        self.drop_last_frame = bool(drop_last_frame)
        self.zz_order = zigzag_order_3d(block_T, block_H, block_W)
        self.clips_per_shard = int(clips_per_shard)

        # Verify count against the first shard (cheap — only one shard read).
        first = torch.load(self._shards[0], map_location='cpu', weights_only=False)
        if len(first) != self.clips_per_shard:
            raise ValueError(
                f"{self._shards[0]} has {len(first)} clips, but config says "
                f"clips_per_shard={self.clips_per_shard}"
            )
        self._cached_shard = first
        self._cached_idx = 0
        self._N = self.clips_per_shard * len(self._shards)

        # In-memory cache of encoded clips: ~100 KB/clip * 10,000 clips ~= 1 GB.
        # First epoch encodes; epoch 2+ reads from RAM only (no shard reload).
        # With num_workers=0 the cache is shared across the whole run (1 GB total).
        # With num_workers>0 each worker keeps its own cache; after a few epochs
        # all workers will have most items, so total RAM ~= num_workers * 1 GB.
        self._mem_cache = {} if cache_in_memory else None

    def __len__(self):
        return self._N

    def _get_shard(self, shard_idx):
        if shard_idx != self._cached_idx:
            self._cached_shard = torch.load(
                self._shards[shard_idx], map_location='cpu', weights_only=False
            )
            self._cached_idx = shard_idx
        return self._cached_shard

    def __getitem__(self, idx):
        if self._mem_cache is not None and idx in self._mem_cache:
            return self._mem_cache[idx]
        shard_idx, local_idx = divmod(idx, self.clips_per_shard)
        shard = self._get_shard(shard_idx)
        sample = shard[local_idx]
        clip = sample['vor']
        if isinstance(clip, torch.Tensor):
            clip = clip.numpy()
        clip = clip.astype(np.float32, copy=False)
        if clip.shape[0] != self.T:
            clip = clip[:self.T]
        tokens = _encode_clip_3d(
            clip, self.block_T, self.block_H, self.block_W,
            self.zz_order, self.low_freqs, self.y_bound,
        )
        out = torch.from_numpy(tokens)
        if self._mem_cache is not None:
            self._mem_cache[idx] = out
        return out


class KarmanVortex3D(DatasetFactory):
    """3D-DCT DCTdiff DatasetFactory for the 2D Karman-vortex simulation.

    Each training sample is a (T, H, W) vorticity clip split into 3D blocks of
    shape (block_T, block_H, block_W). Each block is 3D-DCT-transformed and
    truncated to its first `low_freqs` zigzag (i+j+k ascending) coefficients.

    Output shape: (num_blocks_per_clip, low_freqs * channels).
    """

    def __init__(self, train_dir, test_dir=None, T=200, H=128, W=128,
                 block_T=40, block_H=32, block_W=32, low_freqs=313,
                 Y_bound=None, channels=1, drop_last_frame=True,
                 clips_per_shard=50, cache_in_memory=True, **kwargs):
        super().__init__()
        if Y_bound is None:
            raise ValueError("KarmanVortex3D requires Y_bound (set in stats JSON)")
        if channels != 1:
            raise ValueError(f"KarmanVortex3D expects channels=1, got {channels}")

        assert T % block_T == 0 and H % block_H == 0 and W % block_W == 0, (
            f"({T},{H},{W}) not divisible by ({block_T},{block_H},{block_W})"
        )
        self._tokens = (T // block_T) * (H // block_H) * (W // block_W)
        self._low_freqs = int(low_freqs)
        self.channels = channels

        y_bound = float(np.asarray(Y_bound, dtype=np.float32).reshape(-1)[0])
        train_shards = _list_pt_shards(train_dir, 'shard_')
        self.train = KarmanShardedClipDataset(
            shard_paths=train_shards, T=T,
            block_T=block_T, block_H=block_H, block_W=block_W,
            low_freqs=low_freqs, y_bound=y_bound,
            drop_last_frame=drop_last_frame,
            clips_per_shard=clips_per_shard,
            cache_in_memory=cache_in_memory,
        )
        if test_dir is not None and os.path.isdir(test_dir):
            test_shards = _list_pt_shards(test_dir, 'test_shard_')
            if test_shards:
                self.test = KarmanShardedClipDataset(
                    shard_paths=test_shards, T=T,
                    block_T=block_T, block_H=block_H, block_W=block_W,
                    low_freqs=low_freqs, y_bound=y_bound,
                    drop_last_frame=drop_last_frame,
                    clips_per_shard=clips_per_shard,
                    cache_in_memory=cache_in_memory,
                )

    @property
    def has_label(self):
        return False

    @property
    def fid_stat(self):
        return None

    @property
    def data_shape(self):
        return self._tokens, self._low_freqs * self.channels


class KarmanCachedClipDataset(Dataset):
    """Reads from a precomputed (N_clips, num_blocks, LOW_FREQS) tensor produced
    by precompute_karman_dct.py. Normalisation by y_bound is applied on access
    so y_bound can be tuned without rebuilding the cache."""

    def __init__(self, cache_path, y_bound):
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(
                f"{cache_path} not found — run precompute_karman_dct.py first"
            )
        # mmap the file so we don't need 1 GB resident in every dataloader worker
        try:
            self._data = torch.load(cache_path, map_location='cpu',
                                    weights_only=True, mmap=True)
        except TypeError:
            # older torch — fall back to loading into RAM
            self._data = torch.load(cache_path, map_location='cpu',
                                    weights_only=True)
        self.y_bound = float(y_bound)

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        x = self._data[idx]
        if not torch.is_floating_point(x):
            x = x.float()
        return x.float() / self.y_bound


class KarmanVortex3DCached(DatasetFactory):
    """3D-DCT DCTdiff DatasetFactory backed by precomputed .pt caches.

    Args:
        train_cache: path to karman_dct_train.pt
        test_cache:  path to karman_dct_test.pt (optional)
        T, H, W, block_*, low_freqs, channels: must match the values used by
            precompute_karman_dct.py (shape sanity-checked).
        Y_bound: divisor applied at access time.
    """

    def __init__(self, train_cache, test_cache=None, T=200, H=128, W=128,
                 block_T=40, block_H=32, block_W=32, low_freqs=313,
                 Y_bound=None, channels=1, **kwargs):
        super().__init__()
        if Y_bound is None:
            raise ValueError("KarmanVortex3DCached requires Y_bound")
        if channels != 1:
            raise ValueError(f"KarmanVortex3DCached expects channels=1, got {channels}")

        self._tokens = (T // block_T) * (H // block_H) * (W // block_W)
        self._low_freqs = int(low_freqs)
        self.channels = channels

        y_bound = float(np.asarray(Y_bound, dtype=np.float32).reshape(-1)[0])
        self.train = KarmanCachedClipDataset(train_cache, y_bound)
        # sanity check
        n_blk, lf = self.train._data.shape[1], self.train._data.shape[2]
        if (n_blk, lf) != (self._tokens, self._low_freqs):
            raise ValueError(
                f"cache {train_cache} has shape {tuple(self.train._data.shape)} "
                f"but config expects (N, {self._tokens}, {self._low_freqs})"
            )
        if test_cache and os.path.isfile(test_cache):
            self.test = KarmanCachedClipDataset(test_cache, y_bound)

    @property
    def has_label(self):
        return False

    @property
    def fid_stat(self):
        return None

    @property
    def data_shape(self):
        return self._tokens, self._low_freqs * self.channels


# ---------------------------------------------------------------------------
# 2D Burgers (ux, uy) -- treat ux and uy as independent scalar clips so the
# DCTdiff pipeline is identical to the karman one.
# ---------------------------------------------------------------------------

class BurgersShardedClipDataset(Dataset):
    """Sharded 2D Burgers velocity dataset. Each shard is a list of dicts with
    keys 'ux' and 'uy', each (T, H, W) float32. ux and uy are exposed as
    independent scalar clips, so length = 2 * clips_per_shard * num_shards.
    """

    def __init__(self, shard_paths, T, block_T, block_H, block_W,
                 low_freqs, y_bound, drop_last_frame=True, clips_per_shard=100,
                 cache_in_memory=True):
        if not shard_paths:
            raise FileNotFoundError("no shards found")
        self._shards = list(shard_paths)
        self._cached_idx = -1
        self._cached_shard = None
        self.T = int(T)
        self.block_T = int(block_T)
        self.block_H = int(block_H)
        self.block_W = int(block_W)
        self.low_freqs = int(low_freqs)
        self.y_bound = float(y_bound)
        self.drop_last_frame = bool(drop_last_frame)
        self.zz_order = zigzag_order_3d(block_T, block_H, block_W)
        self.clips_per_shard = int(clips_per_shard)

        # mmap=True keeps tensor storage file-backed (paged via OS page cache,
        # shared across workers) instead of copying ~2.6 GB into per-worker RAM.
        first = torch.load(self._shards[0], map_location='cpu', mmap=True, weights_only=False)
        if len(first) != self.clips_per_shard:
            raise ValueError(
                f"{self._shards[0]} has {len(first)} clips, but config says "
                f"clips_per_shard={self.clips_per_shard}"
            )
        for k in ('ux', 'uy'):
            if k not in first[0]:
                raise KeyError(f"{self._shards[0]}[0] missing field '{k}'")
        self._cached_shard = first
        self._cached_idx = 0
        self._N = 2 * self.clips_per_shard * len(self._shards)

        self._mem_cache = {} if cache_in_memory else None

    def __len__(self):
        return self._N

    def _get_shard(self, shard_idx):
        if shard_idx != self._cached_idx:
            self._cached_shard = torch.load(
                self._shards[shard_idx], map_location='cpu', mmap=True, weights_only=False
            )
            self._cached_idx = shard_idx
        return self._cached_shard

    def __getitem__(self, idx):
        if self._mem_cache is not None and idx in self._mem_cache:
            return self._mem_cache[idx]
        # Layout: consecutive idx pairs (2k, 2k+1) -> (ux, uy) of clip k inside its shard.
        shard_idx, within = divmod(idx, 2 * self.clips_per_shard)
        local_clip, field_idx = divmod(within, 2)
        field = 'ux' if field_idx == 0 else 'uy'
        shard = self._get_shard(shard_idx)
        clip = shard[local_clip][field]
        if isinstance(clip, torch.Tensor):
            clip = clip.numpy()
        clip = clip.astype(np.float32, copy=False)
        if clip.shape[0] != self.T:
            clip = clip[:self.T]
        tokens = _encode_clip_3d(
            clip, self.block_T, self.block_H, self.block_W,
            self.zz_order, self.low_freqs, self.y_bound,
        )
        out = torch.from_numpy(tokens)
        if self._mem_cache is not None:
            self._mem_cache[idx] = out
        return out


class BurgersVortex3D(DatasetFactory):
    """3D-DCT DCTdiff DatasetFactory for the 2D Burgers simulation.

    Each shard dict contains both ux and uy of shape (T, H, W); they are
    treated as separate scalar clips, doubling the effective dataset size.
    """

    def __init__(self, train_dir, test_dir=None, T=200, H=128, W=128,
                 block_T=40, block_H=32, block_W=32, low_freqs=313,
                 Y_bound=None, channels=1, drop_last_frame=True,
                 clips_per_shard=100, cache_in_memory=True, **kwargs):
        super().__init__()
        if Y_bound is None:
            raise ValueError("BurgersVortex3D requires Y_bound (set in stats JSON)")
        if channels != 1:
            raise ValueError(
                f"BurgersVortex3D expects channels=1 (ux/uy treated as separate "
                f"samples), got {channels}"
            )

        assert T % block_T == 0 and H % block_H == 0 and W % block_W == 0, (
            f"({T},{H},{W}) not divisible by ({block_T},{block_H},{block_W})"
        )
        self._tokens = (T // block_T) * (H // block_H) * (W // block_W)
        self._low_freqs = int(low_freqs)
        self.channels = channels

        y_bound = float(np.asarray(Y_bound, dtype=np.float32).reshape(-1)[0])
        train_shards = _list_pt_shards(train_dir, 'shard_')
        self.train = BurgersShardedClipDataset(
            shard_paths=train_shards, T=T,
            block_T=block_T, block_H=block_H, block_W=block_W,
            low_freqs=low_freqs, y_bound=y_bound,
            drop_last_frame=drop_last_frame,
            clips_per_shard=clips_per_shard,
            cache_in_memory=cache_in_memory,
        )
        if test_dir is not None and os.path.isdir(test_dir):
            test_shards = _list_pt_shards(test_dir, 'test_shard_')
            if test_shards:
                self.test = BurgersShardedClipDataset(
                    shard_paths=test_shards, T=T,
                    block_T=block_T, block_H=block_H, block_W=block_W,
                    low_freqs=low_freqs, y_bound=y_bound,
                    drop_last_frame=drop_last_frame,
                    clips_per_shard=clips_per_shard,
                    cache_in_memory=cache_in_memory,
                )

    @property
    def has_label(self):
        return False

    @property
    def fid_stat(self):
        return None

    @property
    def data_shape(self):
        return self._tokens, self._low_freqs * self.channels


class BurgersCachedClipDataset(Dataset):
    """Reads from a precomputed (N_eff_clips, num_blocks, LOW_FREQS) tensor produced
    by precompute_burgers_dct.py. ux/uy of clip k live at rows 2k and 2k+1.
    Normalisation by y_bound is applied on access so y_bound can be tuned without
    rebuilding the cache."""

    def __init__(self, cache_path, y_bound):
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(
                f"{cache_path} not found -- run precompute_burgers_dct.py first"
            )
        try:
            self._data = torch.load(cache_path, map_location='cpu',
                                    weights_only=True, mmap=True)
        except TypeError:
            self._data = torch.load(cache_path, map_location='cpu',
                                    weights_only=True)
        self.y_bound = float(y_bound)

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        x = self._data[idx]
        if not torch.is_floating_point(x):
            x = x.float()
        return x.float() / self.y_bound


class BurgersVortex3DCached(DatasetFactory):
    """3D-DCT DCTdiff DatasetFactory backed by precomputed .pt caches."""

    def __init__(self, train_cache, test_cache=None, T=200, H=128, W=128,
                 block_T=40, block_H=32, block_W=32, low_freqs=204,
                 Y_bound=None, channels=1, **kwargs):
        super().__init__()
        if Y_bound is None:
            raise ValueError("BurgersVortex3DCached requires Y_bound")
        if channels != 1:
            raise ValueError(
                f"BurgersVortex3DCached expects channels=1 (ux/uy as separate samples), got {channels}"
            )

        self._tokens = (T // block_T) * (H // block_H) * (W // block_W)
        self._low_freqs = int(low_freqs)
        self.channels = channels

        y_bound = float(np.asarray(Y_bound, dtype=np.float32).reshape(-1)[0])
        self.train = BurgersCachedClipDataset(train_cache, y_bound)
        n_blk, lf = self.train._data.shape[1], self.train._data.shape[2]
        if (n_blk, lf) != (self._tokens, self._low_freqs):
            raise ValueError(
                f"cache {train_cache} has shape {tuple(self.train._data.shape)} "
                f"but config expects (N, {self._tokens}, {self._low_freqs})"
            )
        if test_cache and os.path.isfile(test_cache):
            self.test = BurgersCachedClipDataset(test_cache, y_bound)

    @property
    def has_label(self):
        return False

    @property
    def fid_stat(self):
        return None

    @property
    def data_shape(self):
        return self._tokens, self._low_freqs * self.channels


# ---------------------------------------------------------------------------
# 2D Burgers + strict t=0-frame conditioning
# ---------------------------------------------------------------------------

class _BurgersT0CondPaired(Dataset):
    """Pairs the 3D-DCT clip cache (N, n_clip_tokens, low_freqs) with the
    2D-DCT t=0 condition cache (N, n_cond_tokens, low_freqs). Returns a single
    concatenated tensor of shape (n_cond_tokens + n_clip_tokens, low_freqs)
    where the first n_cond_tokens rows are the (already-normalised) condition
    and the rest are the (already-normalised) spatiotemporal tokens.
    """

    def __init__(self, clip_cache_path, cond_cache_path, y_bound_clip):
        if not os.path.isfile(clip_cache_path):
            raise FileNotFoundError(clip_cache_path)
        if not os.path.isfile(cond_cache_path):
            raise FileNotFoundError(cond_cache_path)
        try:
            self._clip = torch.load(clip_cache_path, map_location='cpu',
                                    weights_only=True, mmap=True)
            self._cond = torch.load(cond_cache_path, map_location='cpu',
                                    weights_only=True, mmap=True)
        except TypeError:
            self._clip = torch.load(clip_cache_path, map_location='cpu', weights_only=True)
            self._cond = torch.load(cond_cache_path, map_location='cpu', weights_only=True)
        if self._clip.shape[0] != self._cond.shape[0]:
            raise ValueError(
                f'cache size mismatch: clip {self._clip.shape[0]} vs cond {self._cond.shape[0]}'
            )
        if self._clip.shape[2] != self._cond.shape[2]:
            raise ValueError(
                f'low_freqs width mismatch: clip {self._clip.shape[2]} vs cond {self._cond.shape[2]}'
            )
        self.y_bound_clip = float(y_bound_clip)

    def __len__(self):
        return self._clip.shape[0]

    def __getitem__(self, idx):
        clip = self._clip[idx]
        cond = self._cond[idx]
        if not torch.is_floating_point(clip):
            clip = clip.float()
        if not torch.is_floating_point(cond):
            cond = cond.float()
        # The clip cache stores raw (un-normalised) coefs; the cond cache stores
        # already-normalised coefs (precompute_burgers_t0.py bakes in Y_bound_t0).
        clip = clip.float() / self.y_bound_clip
        return torch.cat([cond.float(), clip], dim=0)


class BurgersVortex3DT0CondCached(DatasetFactory):
    """Strict-t=0 conditional DCTdiff factory.

    The model receives a sequence of (n_cond_tokens + n_clip_tokens) tokens; the
    first n_cond_tokens encode the t=0 frame's 2D-DCT (kept clean throughout
    diffusion via LSimple_cond) and the rest are the noisy spatiotemporal
    DCT tokens to be denoised.
    """

    def __init__(self, train_cache, test_cache,
                 train_cond_cache, test_cond_cache,
                 T=200, H=128, W=128, block_T=25, block_H=16, block_W=16,
                 low_freqs=32, Y_bound=None, n_cond_tokens=3,
                 channels=1, **kwargs):
        super().__init__()
        if Y_bound is None:
            raise ValueError("BurgersVortex3DT0CondCached requires Y_bound (clip)")
        if channels != 1:
            raise ValueError(f"expects channels=1, got {channels}")

        self._n_clip_tokens = (T // block_T) * (H // block_H) * (W // block_W)
        self._n_cond_tokens = int(n_cond_tokens)
        self._tokens = self._n_clip_tokens + self._n_cond_tokens
        self._low_freqs = int(low_freqs)
        self.channels = channels

        y_bound = float(np.asarray(Y_bound, dtype=np.float32).reshape(-1)[0])
        self.train = _BurgersT0CondPaired(train_cache, train_cond_cache, y_bound)
        # Sanity: clip token count matches block grid
        n_clip = self.train._clip.shape[1]
        n_cond = self.train._cond.shape[1]
        if n_clip != self._n_clip_tokens or n_cond != self._n_cond_tokens:
            raise ValueError(
                f'cache token counts ({n_clip}, {n_cond}) do not match config '
                f'({self._n_clip_tokens}, {self._n_cond_tokens})'
            )
        if test_cache and test_cond_cache and os.path.isfile(test_cache) and os.path.isfile(test_cond_cache):
            self.test = _BurgersT0CondPaired(test_cache, test_cond_cache, y_bound)

    @property
    def has_label(self):
        return False

    @property
    def fid_stat(self):
        return None

    @property
    def data_shape(self):
        return self._tokens, self._low_freqs * self.channels
