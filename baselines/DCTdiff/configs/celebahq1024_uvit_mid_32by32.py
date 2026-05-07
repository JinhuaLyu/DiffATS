import json
import os

import ml_collections

_CELEBA_HQ_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "CelebA-HQ", "celeba_hq_images")
)
_RGB_STATS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DCT_RGB_STATS", "celebahq1024_32by32_rgb_stats.json")
)


def d(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def _load_rgb_stats():
    rgb_stats_path = os.environ.get("DCTDIFF_RGB_STATS_PATH", _RGB_STATS_PATH)
    if not os.path.exists(rgb_stats_path):
        return {}
    with open(rgb_stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'

    _steps_per_epoch = 30000 // 32  # 937
    _epochs = 1000
    config.train = d(
        n_steps=_steps_per_epoch * _epochs,
        batch_size=32,
        mode='uncond',
        log_interval=500,
        eval_interval=25000,
        save_interval=5000,
        enable_fid=False,
    )

    config.grad_clip = 1.0

    config.optimizer = d(
        name='adamw',
        lr=1e-4,
        weight_decay=0.0,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit',
        tokens=256,
        low_freqs=64,   # 256 * 64 * 12 = 196,608 floats/image
        channels_per_token=12,
        embed_dim=700,
        depth=20,
        num_heads=10,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    rgb_stats = _load_rgb_stats()


    _y_bound = rgb_stats.get('Y_bound_per_freq') or rgb_stats.get('Y_bound', [1996.0])

    config.dataset = d(
        name='celebahq1024',
        path=_CELEBA_HQ_ROOT,
        resolution=1024,
        tokens=256,
        low_freqs=64,
        block_sz=32,
        color_space='rgb',
        Y_bound=_y_bound,
        R_std=rgb_stats.get('R_std'),
        G_std=rgb_stats.get('G_std'),
        B_std=rgb_stats.get('B_std'),
        SNR_scale=12.0,
    )

    config.sample = d(
        sample_steps=250,
        n_samples=10000,
        mini_batch_size=50,
        algorithm='dpm_solver',
        path='samples_celebahq1024',
        save_npz=''
    )

    return config
