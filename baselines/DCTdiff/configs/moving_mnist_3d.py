
import math
import ml_collections


def d(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


# 20k sequences x 100% train split = 20k clips per epoch
N_TRAIN          = 20_000
EPOCHS           = 2000
BATCH_SIZE       = 32

_STEPS_PER_EPOCH = math.ceil(N_TRAIN / BATCH_SIZE)   # 625
_N_STEPS         = EPOCHS * _STEPS_PER_EPOCH           # 1,250,000


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'

    config.train = d(
        n_steps=_N_STEPS,
        batch_size=BATCH_SIZE,
        mode='uncond',
        log_interval=500,
        eval_interval=_STEPS_PER_EPOCH * 5,
        save_interval=_STEPS_PER_EPOCH * 100,
        ckpt_min_step=0,
    )

    config.wandb = d(
        project='DCTdiff-MovingMNIST',
        name=None,
        mode='online',
        tags=('moving_mnist', 'block_4x32x32', 'lf64', 'video'),
    )

    config.optimizer = d(
        name='adam',
        lr=1e-4,
        weight_decay=0.0,
        betas=(0.9, 0.999),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=2_000,
    )

    # UViT for 20-token sequences (block_HW=32 → 2x2 spatial grid per time block)
    config.nnet = d(
        name='uvit',
        tokens=20,
        low_freqs=64,
        channels=1,
        embed_dim=528,
        depth=10,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    # Placeholders — overwritten at runtime from mm_stats_3d.json
    _PLACEHOLDER_STD = [1.0] * (4 * 32 * 32)   # 4096 values

    config.dataset = d(
        name='moving_mnist_3d',
        data_path='/home/bkx8728/Tensor_factor/moving_mnist/moving_mnist_20k_2slow.pt',
        resolution=64,
        channels=1,
        tokens=20,
        low_freqs=64,
        block_T=4,
        block_HW=32,
        Y_bound=[2.0],
        vor_std=_PLACEHOLDER_STD,
        train_frac=1.0,
        SNR_scale=4.0,
        num_workers=4,
    )

    config.sample = d(
        sample_steps=250,
        n_samples=2_048,
        mini_batch_size=64,
        algorithm='euler_maruyama_ode',
        path='./samples/moving_mnist',
        save_npz='',
    )

    return config
