import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import accelerate
import ml_collections
from absl import logging, flags, app
from ml_collections import config_flags
from scipy.fft import dctn, idctn

import sde
import utils
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from DCT_utils import (
    split_clip_into_blocks_3d, combine_blocks_3d,
    zigzag_order_3d, reverse_zigzag_order_3d,
    zigzag_order_2d,
)


STATS_PATH = Path(__file__).parent / 'karman_stats_3d.json'


def _find_latest_ckpt(ckpt_root):
    ckpts = [x for x in os.listdir(ckpt_root) if x.endswith('.ckpt')]
    if not ckpts:
        raise FileNotFoundError(f'no .ckpt under {ckpt_root}')
    steps = [int(x.split('.')[0]) for x in ckpts]
    latest = max(steps)
    return os.path.join(ckpt_root, f'{latest}.ckpt'), latest


def encode_clip_to_tokens(clip, b_T, b_H, b_W, low_freqs, zz):
    blocks = split_clip_into_blocks_3d(clip, b_T, b_H, b_W)
    flat = b_T * b_H * b_W
    dct_blocks = dctn(blocks, type=2, norm='ortho', axes=(1, 2, 3))
    return dct_blocks.reshape(blocks.shape[0], flat)[:, zz][:, :low_freqs].astype(np.float32)


def decode_tokens_to_clip(tokens, b_T, b_H, b_W, low_freqs, rev_zz, T, H, W):
    flat = b_T * b_H * b_W
    full = np.zeros((tokens.shape[0], flat), dtype=np.float32)
    full[:, :low_freqs] = tokens
    full = full[:, rev_zz].reshape(tokens.shape[0], b_T, b_H, b_W)
    decoded = idctn(full.astype(np.float32), type=2, norm='ortho', axes=(1, 2, 3))
    return combine_blocks_3d(decoded, T, H, W, b_T, b_H, b_W)


def encode_t0_cond(t0_frame, cond_dim, zz_2d):
    coefs = dctn(t0_frame, type=2, norm='ortho').reshape(-1)
    return coefs[zz_2d][:cond_dim].astype(np.float32)


def evaluate(config):
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    # ---- load stats ----
    if not STATS_PATH.exists():
        raise FileNotFoundError(STATS_PATH)
    stats = json.loads(STATS_PATH.read_text())
    config.dataset.Y_bound = [stats['Y_bound']]
    config.dataset.vor_std  = stats['vor_std']
    y_bound = float(stats['Y_bound'])
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info')

    # ---- model + EMA ----
    nnet_ema = utils.get_nnet(**config.nnet)
    nnet_ema.to(device).eval()
    ckpt_path, ckpt_step = _find_latest_ckpt(config.ckpt_root)
    logging.info(f'loading EMA from {ckpt_path} (step {ckpt_step})')
    state = torch.load(os.path.join(ckpt_path, 'nnet_ema.pth'), map_location='cpu')
    nnet_ema.load_state_dict(state, strict=True)
    score_model_ema = sde.ScoreModel(
        nnet_ema, pred=config.pred,
        sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale),
    )

    # ---- DCT setup ----
    T = config.dataset.T
    H = config.dataset.H
    W = config.dataset.W
    b_T = config.dataset.block_T
    b_H = config.dataset.block_H
    b_W = config.dataset.block_W
    low_freqs = config.dataset.low_freqs
    cond_dim = int(config.dataset.get('cond_dim', 0))
    zz = zigzag_order_3d(b_T, b_H, b_W)
    rev_zz = reverse_zigzag_order_3d(b_T, b_H, b_W)
    num_blocks = (T // b_T) * (H // b_H) * (W // b_W)
    zz_2d = zigzag_order_2d(H, W) if cond_dim > 0 else None

    # ---- load test clips ----
    test_dir = config.dataset.test_dir
    shards = sorted(glob.glob(os.path.join(test_dir, 'test_shard_*.pt')))
    if not shards:
        raise FileNotFoundError(f'no test_shard_*.pt under {test_dir}')

    test_clips = []
    for sp in shards:
        obj = torch.load(sp, map_location='cpu', weights_only=False)
        for sample in obj:
            v = sample['vor']
            if isinstance(v, torch.Tensor):
                v = v.numpy()
            test_clips.append(v[:T].astype(np.float32))
    n_test = len(test_clips)
    logging.info(f'loaded {n_test} test clips from {len(shards)} shards')

    # ---- compute recons ----
    logging.info('computing recons (DCT-truncation oracle)...')
    recons = []
    for c in test_clips:
        tokens = encode_clip_to_tokens(c, b_T, b_H, b_W, low_freqs, zz)
        recon = decode_tokens_to_clip(tokens, b_T, b_H, b_W, low_freqs, rev_zz, T, H, W)
        recons.append(recon)

    # ---- compute conditioning vectors from t=0 ----
    if cond_dim > 0:
        conds = np.stack([
            encode_t0_cond(c[0], cond_dim, zz_2d) for c in test_clips
        ], axis=0)  # (n_test, cond_dim) — physical scale
        conds = conds / y_bound  # normalize to model space
    else:
        conds = None

    # ---- generate gen (native conditional DPM sampling) ----
    sample_steps = int(config.sample.sample_steps)
    bs = int(config.sample.mini_batch_size)
    snr_scale = float(config.dataset.SNR_scale)

    gens = []
    n_done = 0
    while n_done < n_test:
        b = min(bs, n_test - n_done)
        x_init = torch.randn(b, num_blocks, low_freqs, device=device)
        if cond_dim > 0:
            cond_batch = torch.from_numpy(conds[n_done:n_done + b]).to(device)
            kwargs = dict(cond=cond_batch)
        else:
            kwargs = {}

        noise_schedule = NoiseScheduleVP(schedule='linear', SNR_scale=snr_scale)
        model_fn = model_wrapper(score_model_ema.noise_pred, noise_schedule,
                                 time_input_type='0', model_kwargs=kwargs)
        dpm_solver = DPM_Solver(model_fn, noise_schedule)

        with torch.no_grad():
            samples = dpm_solver.sample(
                x_init, steps=sample_steps,
                eps=1e-4, adaptive_step_size=False, fast_version=True,
            )

        tokens_phys = samples.detach().cpu().numpy() * y_bound
        for i in range(b):
            gen = decode_tokens_to_clip(tokens_phys[i], b_T, b_H, b_W, low_freqs,
                                        rev_zz, T, H, W)
            gens.append(gen)
        n_done += b
        logging.info(f'  generated {n_done}/{n_test}')

    # ---- compute relative-error metrics (L1_rel, L2_rel, rMSE) per clip ----
    eps = 1e-12

    def _metrics(a, b):
        d = a - b
        l1_rel  = float(np.abs(d).sum() / (np.abs(a).sum() + eps))
        l2_rel  = float(np.sqrt(np.sum(d ** 2)) / (np.sqrt(np.sum(a ** 2)) + eps))
        rmse    = float(np.sqrt(np.mean(d ** 2)))
        return l1_rel, l2_rel, rmse

    pairs = {
        'real vs recon': (test_clips, recons),
        'recon vs gen' : (recons,     gens),
        'real vs gen'  : (test_clips, gens),
    }
    results = {}
    for label, (A, B) in pairs.items():
        arr = np.array([_metrics(a, b) for a, b in zip(A, B)])  # (n_test, 3)
        l1_mean, l1_std = float(arr[:, 0].mean()), float(arr[:, 0].std())
        l2_mean, l2_std = float(arr[:, 1].mean()), float(arr[:, 1].std())
        rm_mean, rm_std = float(arr[:, 2].mean()), float(arr[:, 2].std())
        results[label] = dict(
            L1_rel_mean=l1_mean, L1_rel_std=l1_std,
            L2_rel_mean=l2_mean, L2_rel_std=l2_std,
            rMSE_mean=rm_mean,   rMSE_std=rm_std,
        )

    def fmt(v, sig=4):
        return f"{v:.{sig}g}"

    def fmt_pair(m, s):
        return f"{fmt(m, 4)} ± {fmt(s, 2)}"

    header = f"{'Pair':<16} {'L1_rel':<22} {'L2_rel':<22} {'rMSE':<22}"
    print()
    print('=' * 90)
    print(f'Evaluated {n_test} test clips, ckpt step {ckpt_step}, '
          f'conditional={"yes" if cond_dim > 0 else "no"}')
    print('=' * 90)
    print(header)
    for label, r in results.items():
        print(f"{label:<16} "
              f"{fmt_pair(r['L1_rel_mean'], r['L1_rel_std']):<22} "
              f"{fmt_pair(r['L2_rel_mean'], r['L2_rel_std']):<22} "
              f"{fmt_pair(r['rMSE_mean'],   r['rMSE_std']):<22}")
    print('=' * 90)
    # also a clean tab-separated row for the most-relevant pair
    rg = results['real vs gen']
    print()
    print('real vs gen (tab-separated):')
    print(f"{fmt_pair(rg['L1_rel_mean'], rg['L1_rel_std'])}\t"
          f"{fmt_pair(rg['L2_rel_mean'], rg['L2_rel_std'])}\t"
          f"{fmt_pair(rg['rMSE_mean'],   rg['rMSE_std'])}")

    out = {
        'n_test': n_test,
        'ckpt_step': int(ckpt_step),
        'sample_steps': sample_steps,
        'cond_dim': cond_dim,
        'metrics': results,
    }
    eval_path = os.path.join(config.workdir, 'eval_results.json')
    with open(eval_path, 'w') as f:
        json.dump(out, f, indent=2)
    logging.info(f'saved -> {eval_path}')

    # Save the model's generated clips only (real/recon can be recomputed
    # cheaply from the test shards + DCT roundtrip). float16 to halve size.
    # gen[i] corresponds 1:1 to the i-th test clip in glob-sorted order.
    eval_dir = os.path.join(config.workdir, 'eval_clips')
    os.makedirs(eval_dir, exist_ok=True)
    gen_arr = np.stack(gens, axis=0).astype(np.float16)
    torch.save(torch.from_numpy(gen_arr), os.path.join(eval_dir, 'gen.pt'))
    logging.info(f'saved gen tensor float16 shape={gen_arr.shape} '
                 f'({gen_arr.nbytes / (1024 ** 3):.2f} GB) -> {eval_dir}/gen.pt')


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory (must contain ckpts/).")


def main(argv):
    config = FLAGS.config
    if FLAGS.workdir is None:
        raise ValueError("--workdir is required")
    config.workdir   = FLAGS.workdir
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    evaluate(config)


if __name__ == '__main__':
    app.run(main)
