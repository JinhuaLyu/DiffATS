"""
gen_karman_2d.py — Conditional generation from a trained Karman Vortex 2D Tucker
factor-diffusion checkpoint.

For each seed in --seeds, iterate over the full test dataset (500 samples) with
batched DDIM-style sampling (respaced to --sample_steps steps from T_diffusion).
Denormalize the generated Tucker factors and dump one .pt per seed.

Mirrors the sampling call in train_karman_2d.py::generate_and_visualize.
"""

import argparse
import os
import sys
import time

import torch

_EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # exp_karman_vortex
sys.path.insert(0, os.path.join(_EXP, 'train'))
sys.path.insert(0, '${REPO_ROOT}/video')

from diffusion import create_diffusion

from dataset_karman_2d import KarmanTucker2DDataset
from model_karman_2d_dit import (
    build_karman_2d_dit,
    FLAT_MAIN, FLAT_COND,
    FLAT_UT, FLAT_UY, FLAT_G,
    R_T, R_Y, T_DIM, H_DIM,
)


DEFAULT_TRAIN_DIR = ('${DATA_ROOT}/'
                     'tucker_factors/karman_vortex_2d/'
                     'tucker_karman_rT10_rX128_rY30')
DEFAULT_TEST_DIR  = os.path.join(DEFAULT_TRAIN_DIR, 'test_data')
DEFAULT_CKPT      = ('${DATA_ROOT}/'
                     'our_method_results/karman_vortex_2d/checkpoints/'
                     'epoch00200_step0062400.pt')
DEFAULT_OUTDIR    = ('${DATA_ROOT}/'
                     'our_method_generation/karman_vortex_2d')


def pack_test_batch(test_dataset, idxs):
    """Stack normalized tensors for a list of test-dataset indices."""
    samples = [test_dataset[i] for i in idxs]
    U_T   = torch.stack([s['U_T']   for s in samples])
    U_Y   = torch.stack([s['U_Y']   for s in samples])
    G     = torch.stack([s['G']     for s in samples])
    U_ic  = torch.stack([s['U_ic']  for s in samples])
    Vh_ic = torch.stack([s['Vh_ic'] for s in samples])
    niu   = torch.stack([s['niu']   for s in samples])
    cx    = torch.stack([s['cx']    for s in samples])
    cy    = torch.stack([s['cy']    for s in samples])
    r     = torch.stack([s['r']     for s in samples])
    re    = torch.stack([s['Re']    for s in samples])
    sample_idx = torch.tensor([s['idx'] for s in samples], dtype=torch.long)

    x_flat    = torch.cat([U_T.flatten(1), U_Y.flatten(1), G.flatten(1)], dim=1)
    cond_flat = torch.cat([U_ic.flatten(1), Vh_ic.flatten(1)],             dim=1)
    return x_flat, cond_flat, niu, cx, cy, r, re, sample_idx


def unpack_x(x_flat, B):
    c0, c1, c2 = x_flat.split([FLAT_UT, FLAT_UY, FLAT_G], dim=1)
    U_T = c0.reshape(B, T_DIM, R_T)
    U_Y = c1.reshape(B, H_DIM, R_Y)
    G   = c2.reshape(B, R_T, H_DIM, R_Y)
    return U_T, U_Y, G


def load_wrapper_from_ckpt(ckpt_path, device):
    print(f'Loading checkpoint: {ckpt_path}', flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['cfg']
    wrapper = build_karman_2d_dit(cfg).to(device)
    sd = ckpt['ema']
    sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    wrapper.load_state_dict(sd)
    wrapper.eval()
    print(f'  cfg={cfg}  epoch={ckpt.get("epoch")}  step={ckpt.get("step")}',
          flush=True)
    return wrapper, ckpt


@torch.inference_mode()
def generate_one_seed(seed, wrapper, diffusion_sample, test_dataset,
                     batch_size, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    N = len(test_dataset)
    order = list(range(N))

    UT_chunks, UY_chunks, G_chunks, idx_chunks = [], [], [], []
    niu_chunks, cx_chunks, cy_chunks, r_chunks, re_chunks = [], [], [], [], []

    t_seed = time.time()
    for b0 in range(0, N, batch_size):
        b1 = min(b0 + batch_size, N)
        idxs = order[b0:b1]
        B = len(idxs)

        (x_flat, cond_flat, niu, cx, cy, r, re,
         sample_idx) = pack_test_batch(test_dataset, idxs)
        cond_flat = cond_flat.to(device)
        niu = niu.to(device); cx = cx.to(device)
        cy  = cy.to(device);  r  = r.to(device)
        re  = re.to(device)

        noise = torch.randn(B, FLAT_MAIN, device=device)
        samples = diffusion_sample.p_sample_loop(
            wrapper, noise.shape, noise=noise,
            clip_denoised=False,
            model_kwargs={'cond_flat': cond_flat,
                          'niu': niu, 'cx': cx, 'cy': cy, 'r': r, 're': re},
            device=device, progress=False,
        )

        samples = samples.float()
        U_T_n, U_Y_n, G_n = unpack_x(samples, B)
        U_T = test_dataset.denorm(U_T_n, 'UT').cpu()
        U_Y = test_dataset.denorm(U_Y_n, 'UY').cpu()
        G   = test_dataset.denorm(G_n,   'G' ).cpu()

        niu_dn = test_dataset.denorm(niu, 'log_niu').exp().cpu()
        re_dn  = test_dataset.denorm(re,  'log_Re' ).exp().cpu()
        cx_dn  = test_dataset.denorm(cx,  'cx').cpu()
        cy_dn  = test_dataset.denorm(cy,  'cy').cpu()
        r_dn   = test_dataset.denorm(r,   'r' ).cpu()

        UT_chunks.append(U_T); UY_chunks.append(U_Y); G_chunks.append(G)
        idx_chunks.append(sample_idx)
        niu_chunks.append(niu_dn); re_chunks.append(re_dn)
        cx_chunks.append(cx_dn); cy_chunks.append(cy_dn); r_chunks.append(r_dn)

        print(f'  [seed {seed}]  batch {b0:4d}:{b1:4d}  '
              f'({b1-b0} samples)  elapsed={time.time()-t_seed:.1f}s',
              flush=True)

    out = {
        'U_T'       : torch.cat(UT_chunks,  dim=0),
        'U_Y'       : torch.cat(UY_chunks,  dim=0),
        'G'         : torch.cat(G_chunks,   dim=0),
        'sample_idx': torch.cat(idx_chunks, dim=0),
        'niu'       : torch.cat(niu_chunks, dim=0),
        'Re'        : torch.cat(re_chunks,  dim=0),
        'cx'        : torch.cat(cx_chunks,  dim=0),
        'cy'        : torch.cat(cy_chunks,  dim=0),
        'r'         : torch.cat(r_chunks,   dim=0),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',           type=str, default=DEFAULT_CKPT)
    parser.add_argument('--output_dir',     type=str, default=DEFAULT_OUTDIR)
    parser.add_argument('--train_data_dir', type=str, default=DEFAULT_TRAIN_DIR)
    parser.add_argument('--test_data_dir',  type=str, default=DEFAULT_TEST_DIR)
    parser.add_argument('--batch_size',     type=int, default=50)
    parser.add_argument('--seeds',          type=int, nargs='+',
                        default=[0, 1, 2, 3, 4])
    parser.add_argument('--sample_steps',   type=int, default=250)
    parser.add_argument('--noise_schedule', type=str, default='linear')
    parser.add_argument('--T_diffusion',    type=int, default=1000)
    parser.add_argument('--device',         type=str, default='cuda:0')
    parser.add_argument('--epoch_tag',      type=str, default='epoch00200',
                        help='filename prefix; change if using other ckpts')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else 'cpu')
    print(f'Device: {device}', flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_dataset = KarmanTucker2DDataset(
        args.train_data_dir, split='all', device=device,
    )
    test_dataset = KarmanTucker2DDataset(
        args.test_data_dir, split='all', device=device,
        external_stats=train_dataset.stats,
    )
    print(f'Test samples: {len(test_dataset)}', flush=True)
    assert len(test_dataset) == 500, \
        f'Expected 500 test samples, got {len(test_dataset)}'

    wrapper, ckpt = load_wrapper_from_ckpt(args.ckpt, device)

    diffusion_sample = create_diffusion(
        timestep_respacing=str(args.sample_steps),
        noise_schedule=args.noise_schedule,
        learn_sigma=False,
        diffusion_steps=args.T_diffusion,
    )

    print(f'Sample steps: {args.sample_steps} (respaced from '
          f'{args.T_diffusion})  schedule={args.noise_schedule}', flush=True)

    t_total = time.time()
    for seed in args.seeds:
        print(f'\n===== seed {seed} =====', flush=True)
        t0 = time.time()
        out = generate_one_seed(
            seed, wrapper, diffusion_sample, test_dataset,
            args.batch_size, device,
        )
        out.update({
            'seed'          : seed,
            'epoch'         : int(ckpt.get('epoch', -1)),
            'step'          : int(ckpt.get('step', -1)),
            'ckpt_path'     : args.ckpt,
            'sample_steps'  : args.sample_steps,
            'noise_schedule': args.noise_schedule,
            'T_diffusion'   : args.T_diffusion,
        })

        out_path = os.path.join(args.output_dir,
                                f'{args.epoch_tag}_seed{seed}.pt')
        torch.save(out, out_path)
        print(f'  -> saved {out_path}  '
              f'(U_T={tuple(out["U_T"].shape)}, '
              f'U_Y={tuple(out["U_Y"].shape)}, '
              f'G={tuple(out["G"].shape)})  '
              f'seed_time={time.time()-t0:.1f}s',
              flush=True)

    print(f'\nAll seeds done in {time.time()-t_total:.1f}s', flush=True)


if __name__ == '__main__':
    main()
