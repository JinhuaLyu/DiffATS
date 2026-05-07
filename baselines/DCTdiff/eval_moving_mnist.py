import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import utils
import sde
from DCT_utils import reverse_zigzag_order_3d
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from configs.moving_mnist import get_config


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',  required=True, type=str,
                    help='Path to e.g. .../ckpts/1250000.ckpt')
    ap.add_argument('--out',   required=True, type=str)
    ap.add_argument('--n',     type=int, default=10000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--steps', type=int, default=250)
    ap.add_argument('--seed',  type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    print(f'device: {device}', flush=True)

    # -------- Load config and override Y_bound / vor_std from JSON stats --------
    config = get_config()
    stats_path = THIS_DIR / 'mm_stats_3d.json'
    stats = json.loads(stats_path.read_text())
    config.dataset.Y_bound = [stats['Y_bound']]
    config.dataset.vor_std = stats['vor_std']
    Y_bound = config.dataset.Y_bound

    # -------- Build the network (just one copy — load EMA weights into it) --------
    nnet = utils.get_nnet(**config.nnet).to(device)
    nnet.eval()
    ema_path = os.path.join(args.ckpt, 'nnet_ema.pth')
    print(f'loading EMA weights: {ema_path}', flush=True)
    state = torch.load(ema_path, map_location=device)
    nnet.load_state_dict(state)

    # -------- Wrap in ScoreModel + DPM-Solver --------
    score_model = sde.ScoreModel(
        nnet, pred=config.pred,
        sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale),
    )
    noise_schedule = NoiseScheduleVP(
        schedule='linear',
        SNR_scale=config.dataset.SNR_scale,
    )
    model_fn = model_wrapper(
        score_model.noise_pred, noise_schedule, time_input_type='0',
    )
    dpm_solver = DPM_Solver(model_fn, noise_schedule)

    # -------- Decoder constants --------
    bT = config.dataset.block_T
    bHW = config.dataset.block_HW
    reverse_order = reverse_zigzag_order_3d(bT, bHW, bHW)
    T_full = bT * (config.dataset.tokens // ((config.dataset.resolution // bHW) ** 2))
    H = W = config.dataset.resolution
    data_shape = (config.dataset.tokens, config.dataset.low_freqs * config.dataset.channels)
    print(f'data_shape={data_shape}  T={T_full} H={H} W={W}  bT={bT} bHW={bHW}', flush=True)
    print(f'sampling N={args.n}  batch={args.batch}  steps={args.steps}', flush=True)

    # -------- Loop: sample → decode → uint8 → accumulate --------
    N = args.n
    out = np.empty((N, T_full, H, W), dtype=np.uint8)
    idx = 0
    t0 = time.time()
    while idx < N:
        b = min(args.batch, N - idx)
        x_init = torch.randn(b, *data_shape, device=device)
        samples = dpm_solver.sample(
            x_init, steps=args.steps,
            eps=1e-4, adaptive_step_size=False, fast_version=True,
        )  # (b, tokens, low_freqs)

        clips = utils.DCT3D_samples_to_clips(
            samples.detach().cpu().numpy(),
            tokens=config.dataset.tokens,
            low_freqs=config.dataset.low_freqs,
            block_T=bT, block_HW=bHW,
            reverse_order_3d=reverse_order,
            T=T_full, H=H, W=W,
            Y_bound=Y_bound,
        )  # (b, T, H, W) float32, pixel scale

        clips = np.clip(clips, 0.0, 255.0).round().astype(np.uint8)
        out[idx:idx + b] = clips
        idx += b
        if (idx // args.batch) % 10 == 0 or idx == N:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0.0
            print(f'  sampled {idx:6d}/{N}   '
                  f'elapsed={elapsed:7.1f}s   {rate:.2f} clips/s', flush=True)

    print(f'done sampling in {time.time() - t0:.1f}s', flush=True)
    print(f'output shape: {out.shape}  dtype: {out.dtype}  '
          f'min={out.min()} max={out.max()}', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(torch.from_numpy(out), args.out)
    print(f'saved: {args.out}', flush=True)


if __name__ == '__main__':
    main()
