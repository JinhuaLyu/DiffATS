"""Strict t=0-frame conditional DCTdiff config for 2D Burgers (block-5 variant).

Forecasting: condition on the single t=0 frame (encoded via 2D DCT into 4
condition tokens of width 26, ~0.63% of t=0 pixels); generate the full
200-frame clip whose t=0 must match.

Larger spatial blocks (32x32 → 4x4=16 spatial regions per frame) eliminate
the spatial blockiness seen with the (25,16,16) variant; finer temporal
blocks (block_T=5 → 40 temporal blocks) make the seq long enough to fill
the ~8h training budget at the same ~42M-param model.

Sequence layout fed to UViT:
    [4 condition tokens (clean)]   ||   [640 spatiotemporal tokens (denoise)]
    n_cond_tokens = 4, treated as observed in LSimple_cond — clean at every
    diffusion step, loss only on the 640 spatiotemporal tokens.

Tokenisation
    clip = (T=200, H=128, W=128) -> block (5,32,32) -> 40*4*4 = 640 tokens, low_freqs=26
    t=0 condition = 2D DCT of the (128,128) initial frame
                  -> first 104 zigzag coefs -> reshape to 4 tokens of 26
    total seq    = 4 + 640 = 644 tokens
    total kept   = 16,640 spatiotemporal + 104 cond = 16,744 floats per clip
                   (~197:1 compression vs 201*128*128 = 3,293,184 voxels)
"""

import math
import ml_collections


def d(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


N_TRAIN          = 10_000
EPOCHS           = 500
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
        mode='cond_t0',
        log_interval=100,
        eval_interval=_N_STEPS + 1,   # disabled; rely on output.log loss curve
        save_interval=_STEPS_PER_EPOCH * 50,
        ckpt_min_step=0,
    )

    config.optimizer = d(
        name='adam', lr=1e-4, weight_decay=0.0, betas=(0.9, 0.999),
    )
    config.lr_scheduler = d(name='customized', warmup_steps=2_000)

    # UViT seq length 644 (4 cond + 640 spatiotemporal); same ~42M-param model
    # as the prior strict-t=0 run, but per-step time grows ~1.5x to fill ~9h.
    config.nnet = d(
        name='uvit',
        tokens=644,
        low_freqs=32,
        channels=1,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=3.6,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    _PLACEHOLDER_STD = [1.0] * (5 * 32 * 32)  # 5,120

    config.dataset = d(
        name='burgers_2d_t0cond_cached',
        train_cache='/scratch/bkx8728/burgers_dctdiff_runs/burgers_dct_cache_b5/burgers_dct_train.pt',
        test_cache='/scratch/bkx8728/burgers_dctdiff_runs/burgers_dct_cache_b5/burgers_dct_test.pt',
        train_cond_cache='/scratch/bkx8728/burgers_dctdiff_runs/burgers_dct_cache_b5/burgers_t0_cond_train.pt',
        test_cond_cache='/scratch/bkx8728/burgers_dctdiff_runs/burgers_dct_cache_b5/burgers_t0_cond_test.pt',
        T=200, H=128, W=128,
        block_T=25, block_H=16, block_W=16,
        low_freqs=32,
        channels=1,
        Y_bound=[1.0],            # overwritten from JSON
        vor_std=_PLACEHOLDER_STD, # overwritten from JSON
        SNR_scale=4.0,
        num_workers=4,
        drop_last_frame=True,
        clips_per_shard=100,
        cache_in_memory=True,
        # Strict-t=0 conditioning: first 4 tokens are the 2D-DCT of t=0,
        # always clean at every diffusion step.
        n_cond_tokens=4,
    )

    config.sample = d(
        sample_steps=250, n_samples=64, mini_batch_size=8,
        algorithm='euler_maruyama_ode',
        path='./samples/burgers_2d_t0cond', save_npz='',
    )

    return config
