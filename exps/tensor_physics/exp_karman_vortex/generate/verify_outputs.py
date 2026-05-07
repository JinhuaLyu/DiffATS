"""
verify_outputs.py — Sanity check the generated seed files.

For each seed:
  1. Load .pt, assert shapes.
  2. For N=8 samples: reconstruct video from generated factors and compare
     against the ground-truth video reconstructed from test_dataset.
  3. Check that different seeds give different factors (noise diversity).
"""
import argparse
import os
import sys

import numpy as np
import torch

_EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_EXP, 'train'))
sys.path.insert(0, '/home/x-jlyu5/jinhua/factor_diffusion/video')

from dataset_karman_2d import KarmanTucker2DDataset, reconstruct_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True,
                        help='Directory containing epoch{:05d}_seed{}.pt')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--n_check', type=int, default=8)
    parser.add_argument('--train_data_dir', type=str,
                        default=('/anvil/projects/x-eng260004/factor_diffusion/'
                                 'tucker_factors/karman_vortex_2d/'
                                 'tucker_karman_rT10_rX128_rY30'))
    parser.add_argument('--test_data_dir', type=str,
                        default=('/anvil/projects/x-eng260004/factor_diffusion/'
                                 'tucker_factors/karman_vortex_2d/'
                                 'tucker_karman_rT10_rX128_rY30/test_data'))
    args = parser.parse_args()

    tag = f'epoch{args.epoch:05d}'
    files = [os.path.join(args.dir, f'{tag}_seed{s}.pt') for s in args.seeds]
    for f in files:
        assert os.path.exists(f), f'Missing: {f}'
    print(f'All {len(files)} seed files found.')

    # ── Shape check ─────────────────────────────────────────────────────────
    outs = {}
    for s, f in zip(args.seeds, files):
        d = torch.load(f, map_location='cpu', weights_only=False)
        assert d['U_T'].shape == (500, 200, 10), d['U_T'].shape
        assert d['U_Y'].shape == (500, 128, 30), d['U_Y'].shape
        assert d['G'].shape   == (500, 10, 128, 30), d['G'].shape
        assert d['sample_idx'].shape == (500,), d['sample_idx'].shape
        print(f'  seed {s}: shapes OK  '
              f'(epoch={d.get("epoch")}, step={d.get("step")}, '
              f'sample_steps={d.get("sample_steps")})')
        outs[s] = d

    # ── Seeds differ ────────────────────────────────────────────────────────
    s0, s1 = args.seeds[0], args.seeds[1]
    diff = (outs[s0]['U_T'] - outs[s1]['U_T']).abs().mean().item()
    print(f'  Mean |U_T diff| between seed {s0} and seed {s1}: {diff:.4f} '
          f'(should be > 0)')
    assert diff > 1e-4, 'Seeds produced identical outputs!'

    # ── Video rel-err vs ground truth ───────────────────────────────────────
    print(f'\nReconstructing {args.n_check} videos and computing rel-err vs GT...')
    train_dataset = KarmanTucker2DDataset(args.train_data_dir, split='all',
                                            device='cpu')
    test_dataset  = KarmanTucker2DDataset(args.test_data_dir, split='all',
                                            device='cpu',
                                            external_stats=train_dataset.stats)

    for s in args.seeds:
        d = outs[s]
        rel_errs = []
        for i in range(args.n_check):
            ti = int(d['sample_idx'][i].item())
            U_T_gen = d['U_T'][i].numpy().astype(np.float32)
            U_Y_gen = d['U_Y'][i].numpy().astype(np.float32)
            G_gen   = d['G'][i].numpy().astype(np.float32)
            v_gen   = reconstruct_video(U_T_gen, U_Y_gen, G_gen)

            U_T_gt  = test_dataset.UT_all[ti].cpu().numpy().astype(np.float32)
            U_Y_gt  = test_dataset.UY_all[ti].cpu().numpy().astype(np.float32)
            G_gt    = test_dataset.G_all[ti].cpu().numpy().astype(np.float32)
            v_gt    = reconstruct_video(U_T_gt, U_Y_gt, G_gt)

            e = float(np.linalg.norm(v_gen - v_gt) /
                      (np.linalg.norm(v_gt) + 1e-8))
            rel_errs.append(e)
        m = float(np.mean(rel_errs))
        print(f'  seed {s}: rel-err {rel_errs} mean={m:.4f}')

    print('\nVerification PASSED.')


if __name__ == '__main__':
    main()
