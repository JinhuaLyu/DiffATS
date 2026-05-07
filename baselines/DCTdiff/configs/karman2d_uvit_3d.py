"""DCTdiff config for the 2D Karman vortex (sharded, single-channel vorticity).

Tokenisation
    clip = (T=200, H=128, W=128)  (drop last frame from 201)
    block (40, 32, 32) -> (T/40)*(H/32)*(W/32) = 5*4*4 = 80 tokens
    low_freqs = 313  -> keep the 313 lowest 3D zigzag coeffs per block
    channels  = 1
    total kept coefs = 80 * 313 = 25,040 per clip
    data_shape = (80, 313)

Stats are produced by karman_3d_statis.py -> karman_stats_3d.json (overrides
Y_bound and vor_std at runtime).

Model: UViT (embed_dim=512, depth=12, num_heads=8, mlp_ratio=3.5)
    -> ~41M parameters for sequence length 80.
"""

import math
import ml_collections


def d(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


# 200 train shards * 50 clips/shard = 10_000 clips
N_TRAIN          = 10_000
EPOCHS           = 1000
BATCH_SIZE       = 32

_STEPS_PER_EPOCH = math.ceil(N_TRAIN / BATCH_SIZE)
_N_STEPS         = EPOCHS * _STEPS_PER_EPOCH


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'

    config.train = d(
        n_steps=_N_STEPS,
        batch_size=BATCH_SIZE,
        mode='uncond',
        log_interval=100,
        eval_interval=_STEPS_PER_EPOCH * 5,
        save_interval=_STEPS_PER_EPOCH * 50,
        ckpt_min_step=0,
    )

    config.optimizer = d(
        name='adamw',
        lr=1e-4,
        weight_decay=0.0,
        betas=(0.9, 0.999),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=2_000,
    )

    # UViT — sequence length 256 (block (50, 16, 16) = 4*8*8). ~41.6M params,
    # plus a t=0 conditioning vector of dim 1024 (2D-DCT zigzag low-freq of
    # the first frame) projected via 2-layer MLP into one prepended token.
    config.nnet = d(
        name='uvit',
        tokens=256,
        low_freqs=98,
        channels=1,
        cond_dim=1024,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=3.6,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    # Placeholders — overwritten at runtime from karman_stats_3d.json.
    _PLACEHOLDER_STD = [1.0] * (50 * 16 * 16)  # 12,800

    # Default: read precomputed truncated DCT cache (~1 GB total). Built by
    # precompute_karman_dct.py. The raw-shard mode (`name='karman_vortex_3d'`)
    # is incompatible with shuffle=True + multi-worker dataloading on this
    # dataset: each worker thrashes 660 MB shards and per-worker in-memory
    # caches multiply, OOM-ing the 64 GB SLURM allocation.
    config.dataset = d(
        name='karman_vortex_3d_cached',
        train_cache='/projects/p32954/bkx8728/karman_vortex_2d/dct_cache/karman_dct_train.pt',
        test_cache='/projects/p32954/bkx8728/karman_vortex_2d/dct_cache/karman_dct_test.pt',
        train_cond='/projects/p32954/bkx8728/karman_vortex_2d/dct_cache/karman_t0_cond_train.pt',
        test_cond='/projects/p32954/bkx8728/karman_vortex_2d/dct_cache/karman_t0_cond_test.pt',
        cond_dim=1024,
        # raw-shard fallback paths (only used when name='karman_vortex_3d')
        train_dir='/projects/p32954/bkx8728/karman_vortex_2d',
        test_dir='/projects/p32954/bkx8728/karman_vortex_2d/test_data',
        T=200,
        H=128,
        W=128,
        block_T=50,
        block_H=16,
        block_W=16,
        low_freqs=98,
        channels=1,
        Y_bound=[1.0],            # overwritten from JSON
        vor_std=_PLACEHOLDER_STD, # overwritten from JSON
        SNR_scale=4.0,
        num_workers=4,
        drop_last_frame=True,
        clips_per_shard=50,
        cache_in_memory=True,
    )

    config.sample = d(
        sample_steps=250,
        n_samples=1000,
        mini_batch_size=32,
        algorithm='euler_maruyama_ode',
        path='./samples/karman_vortex',
        save_npz='',
    )

    return config
