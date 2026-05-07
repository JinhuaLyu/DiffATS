import ml_collections


def d(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


# Placeholder (128 = M_x * low_freqs = 4 * 32); use_reweight=False so unused.
# Re-measure on dataset if reweighting is ever re-enabled.
_PER_FREQ_STD = [1.0] * 128


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'

    # B_t=16, B_x=32 blocks: 16x compression, 120 tokens (half of v2's 240).
    # Attention cost drops 4x -> 1000 epochs fits in ~1-2 h on H100 bf16.
    # Larger blocks (16x32 vs 8x16) halve the visible seams in both t and x.
    config.train = d(
        n_epochs=1000,
        n_steps=320_000,
        batch_size=32,
        mixed_precision='bf16',
        mode='cond',
        use_reweight=False,
        log_interval=500,
        eval_interval=20_000,
        save_interval=40_000,
    )

    config.optimizer = d(
        name='adamw',
        lr=1e-4,
        weight_decay=0.0,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=2000,
    )


    config.nnet = d(
        name='uvit_reaction',
        tokens=120,
        low_freqs=32,
        M_x=4,
        embed_dim=792,
        depth=14,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        nu_log_scale=True,
        nu_log_min=-5.0,
        nu_log_max=-1.0,
        rho_log_scale=True,
        rho_log_min=-1.0,
        rho_log_max=0.301,            # log10(2.0)
    )

    config.dataset = d(
        name='reaction1d',
        path='/scratch/bkx8728/reaction_1d/reaction_1d_train.pt',
        test_path='/scratch/bkx8728/reaction_1d/reaction_1d_test.pt',
        T_pad=240,
        ic_repeat=8,
        gen_start=32,                 # 2 * B_t = 2 * 16
        ic_block_rows=2,
        B_t=16,
        B_x=32,
        M_x=4,
        low_freqs=32,
        shift=0.5,                    # field is bounded in [0, 1]
        u_scale=0.2541198134,
        Y_bound=4.22,
        per_freq_std=_PER_FREQ_STD,
        SNR_scale=1.0,
        n_ic_tokens=16,               # = ic_block_rows * n_x_macro = 2 * 8
    )

    config.sample = d(
        sample_steps=250,
        n_samples=500,
        mini_batch_size=32,
        algorithm='euler_maruyama_ode',
    )

    return config
