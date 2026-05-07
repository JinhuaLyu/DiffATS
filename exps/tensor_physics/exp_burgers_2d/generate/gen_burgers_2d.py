"""
gen_burgers_2d.py — Conditional generation from a trained Burgers 2D Tucker
factor-diffusion checkpoint.

For each seed in --seeds, iterate over the full test dataset (500 samples) with
batched DDIM-style sampling (respaced to --sample_steps steps from T_diffusion).
Denormalize the generated Tucker factors and dump one .pt per seed.

Mirrors the sampling call in train_burgers_2d.py::generate_and_visualize.
"""

import argparse
import glob
import os
import sys
import time

import torch

_EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # exp_burgers_2d
sys.path.insert(0, os.path.join(_EXP, 'train'))
sys.path.insert(0, '/home/x-jlyu5/jinhua/factor_diffusion/video')

from diffusion import create_diffusion

from dataset_burgers_2d import BurgersTucker2DDataset
from model_burgers_2d_dit import (
    build_burgers_2d_dit,
    FLAT_MAIN, FLAT_COND,
    FLAT_U1, FLAT_U3, FLAT_G,
    R_T, R_W, T_DIM, H_DIM,
)


DEFAULT_TRAIN_DIR = ('/anvil/projects/x-eng260004/factor_diffusion/'
                     'tucker_factors/burgers_2d/'
                     'tucker_burgers_rT5_rH20_rW20')
DEFAULT_TEST_DIR  = os.path.join(DEFAULT_TRAIN_DIR, 'test_data')
DEFAULT_CKPT_DIR  = ('/anvil/projects/x-eng260004/factor_diffusion/'
                     'our_method_results/burgers_2d/checkpoints')
DEFAULT_OUTDIR    = ('/anvil/projects/x-eng260004/factor_diffusion/'
                     'our_method_generation/burgers_2d')


def resolve_ckpt(ckpt_arg, ckpt_dir, epoch):
    """If ckpt_arg given, use it; else auto-discover epoch{epoch:05d}_*.pt."""
    if ckpt_arg:
        return ckpt_arg
    pattern = os.path.join(ckpt_dir, f'epoch{epoch:05d}_step*.pt')
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f'No checkpoint matching {pattern}. Pass --ckpt explicitly.'
        )
    return matches[-1]


def pack_test_batch(test_dataset, idxs):
    samples = [test_dataset[i] for i in idxs]
    U1    = torch.stack([s['U1']    for s in samples])
    U3    = torch.stack([s['U3']    for s in samples])
    G     = torch.stack([s['G']     for s in samples])
    U_ic  = torch.stack([s['U_ic']  for s in samples])
    Vh_ic = torch.stack([s['Vh_ic'] for s in samples])
    nu    = torch.stack([s['nu']    for s in samples])
    cd    = torch.stack([s['cd']    for s in samples])
    sample_idx = torch.tensor([s['idx'] for s in samples], dtype=torch.long)

    x_flat    = torch.cat([U1.flatten(1), U3.flatten(1), G.flatten(1)], dim=1)
    cond_flat = torch.cat([U_ic.flatten(1), Vh_ic.flatten(1)],           dim=1)
    return x_flat, cond_flat, nu, cd, sample_idx


def unpack_x(x_flat, B):
    c0, c1, c2 = x_flat.split([FLAT_U1, FLAT_U3, FLAT_G], dim=1)
    U1 = c0.reshape(B, T_DIM, R_T)
    U3 = c1.reshape(B, H_DIM, R_W)
    G  = c2.reshape(B, R_T, H_DIM, R_W)
    return U1, U3, G


def load_wrapper_from_ckpt(ckpt_path, device):
    print(f'Loading checkpoint: {ckpt_path}', flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['cfg']
    wrapper = build_burgers_2d_dit(cfg).to(device)
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

    U1_chunks, U3_chunks, G_chunks, idx_chunks = [], [], [], []
    nu_chunks, cd_chunks = [], []

    t_seed = time.time()
    for b0 in range(0, N, batch_size):
        b1 = min(b0 + batch_size, N)
        idxs = order[b0:b1]
        B = len(idxs)

        (_x_flat, cond_flat, nu, cd, sample_idx) = pack_test_batch(
            test_dataset, idxs,
        )
        cond_flat = cond_flat.to(device)
        nu = nu.to(device); cd = cd.to(device)

        noise = torch.randn(B, FLAT_MAIN, device=device)
        samples = diffusion_sample.p_sample_loop(
            wrapper, noise.shape, noise=noise,
            clip_denoised=False,
            model_kwargs={'cond_flat': cond_flat, 'nu': nu, 'cd': cd},
            device=device, progress=False,
        )

        samples = samples.float()
        U1_n, U3_n, G_n = unpack_x(samples, B)
        U1 = test_dataset.denorm(U1_n, 'U1').cpu()
        U3 = test_dataset.denorm(U3_n, 'U3').cpu()
        G  = test_dataset.denorm(G_n,  'G' ).cpu()

        nu_dn = test_dataset.denorm(nu, 'log_nu').exp().cpu()
        cd_dn = test_dataset.denorm(cd, 'cd').cpu()

        U1_chunks.append(U1); U3_chunks.append(U3); G_chunks.append(G)
        idx_chunks.append(sample_idx)
        nu_chunks.append(nu_dn); cd_chunks.append(cd_dn)

        print(f'  [seed {seed}]  batch {b0:4d}:{b1:4d}  '
              f'({b1-b0} samples)  elapsed={time.time()-t_seed:.1f}s',
              flush=True)

    out = {
        'U1'        : torch.cat(U1_chunks, dim=0),
        'U3'        : torch.cat(U3_chunks, dim=0),
        'G'         : torch.cat(G_chunks,  dim=0),
        'sample_idx': torch.cat(idx_chunks, dim=0),
        'nu'        : torch.cat(nu_chunks, dim=0),
        'cd'        : torch.cat(cd_chunks, dim=0),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',           type=str, default=None,
                        help='Explicit checkpoint path. If None, auto-discover '
                             'by --epoch inside --ckpt_dir.')
    parser.add_argument('--ckpt_dir',       type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument('--epoch',          type=int, default=200,
                        help='Epoch number to auto-discover if --ckpt not set.')
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
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else 'cpu')
    print(f'Device: {device}', flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ckpt_path = resolve_ckpt(args.ckpt, args.ckpt_dir, args.epoch)
    epoch_tag = f'epoch{args.epoch:05d}'

    train_dataset = BurgersTucker2DDataset(
        args.train_data_dir, split='all', device=device,
    )
    test_dataset = BurgersTucker2DDataset(
        args.test_data_dir, split='all', device=device,
        external_stats=train_dataset.stats,
    )
    print(f'Test samples: {len(test_dataset)}  '
          '(burgers stores ux+uy as separate rows)', flush=True)

    wrapper, ckpt = load_wrapper_from_ckpt(ckpt_path, device)

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
            'ckpt_path'     : ckpt_path,
            'sample_steps'  : args.sample_steps,
            'noise_schedule': args.noise_schedule,
            'T_diffusion'   : args.T_diffusion,
        })

        out_path = os.path.join(args.output_dir,
                                f'{epoch_tag}_seed{seed}.pt')
        torch.save(out, out_path)
        print(f'  -> saved {out_path}  '
              f'(U1={tuple(out["U1"].shape)}, '
              f'U3={tuple(out["U3"].shape)}, '
              f'G={tuple(out["G"].shape)})  '
              f'seed_time={time.time()-t0:.1f}s',
              flush=True)
        del out

    print(f'\nAll seeds done in {time.time()-t_total:.1f}s', flush=True)


if __name__ == '__main__':
    main()
