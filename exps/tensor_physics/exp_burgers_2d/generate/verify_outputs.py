"""
verify_outputs.py — Sanity check the generated Burgers seed files.
"""
import argparse
import os
import sys

import numpy as np
import torch

_EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_EXP, 'train'))
sys.path.insert(0, '/home/x-jlyu5/jinhua/factor_diffusion/video')

from dataset_burgers_2d import BurgersTucker2DDataset, reconstruct_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--n_check', type=int, default=8)
    parser.add_argument('--train_data_dir', type=str,
                        default=('/anvil/projects/x-eng260004/factor_diffusion/'
                                 'tucker_factors/burgers_2d/'
                                 'tucker_burgers_rT5_rH20_rW20'))
    parser.add_argument('--test_data_dir', type=str,
                        default=('/anvil/projects/x-eng260004/factor_diffusion/'
                                 'tucker_factors/burgers_2d/'
                                 'tucker_burgers_rT5_rH20_rW20/test_data'))
    args = parser.parse_args()

    tag = f'epoch{args.epoch:05d}'
    files = [os.path.join(args.dir, f'{tag}_seed{s}.pt') for s in args.seeds]
    for f in files:
        assert os.path.exists(f), f'Missing: {f}'
    print(f'All {len(files)} seed files found.')

    outs = {}
    for s, f in zip(args.seeds, files):
        d = torch.load(f, map_location='cpu', weights_only=False)
        N = d['U1'].shape[0]
        assert d['U1'].shape == (N, 200, 5), d['U1'].shape
        assert d['U3'].shape == (N, 128, 20), d['U3'].shape
        assert d['G'].shape  == (N, 5, 128, 20), d['G'].shape
        assert d['sample_idx'].shape == (N,), d['sample_idx'].shape
        print(f'  seed {s}: N={N}  shapes OK  '
              f'(epoch={d.get("epoch")}, step={d.get("step")}, '
              f'sample_steps={d.get("sample_steps")})')
        outs[s] = d

    s0, s1 = args.seeds[0], args.seeds[1]
    diff = (outs[s0]['U1'] - outs[s1]['U1']).abs().mean().item()
    print(f'  Mean |U1 diff| between seed {s0} and seed {s1}: {diff:.4f} '
          f'(should be > 0)')
    assert diff > 1e-4, 'Seeds produced identical outputs!'

    print(f'\nReconstructing {args.n_check} videos and computing rel-err vs GT...')
    train_dataset = BurgersTucker2DDataset(args.train_data_dir, split='all',
                                            device='cpu')
    test_dataset  = BurgersTucker2DDataset(args.test_data_dir, split='all',
                                            device='cpu',
                                            external_stats=train_dataset.stats)

    for s in args.seeds:
        d = outs[s]
        rel_errs = []
        for i in range(args.n_check):
            ti = int(d['sample_idx'][i].item())
            U1_gen = d['U1'][i].numpy().astype(np.float32)
            U3_gen = d['U3'][i].numpy().astype(np.float32)
            G_gen  = d['G'][i].numpy().astype(np.float32)
            v_gen  = reconstruct_video(U1_gen, U3_gen, G_gen)

            U1_gt  = test_dataset.U1_all[ti].cpu().numpy().astype(np.float32)
            U3_gt  = test_dataset.U3_all[ti].cpu().numpy().astype(np.float32)
            G_gt   = test_dataset.G_all[ti].cpu().numpy().astype(np.float32)
            v_gt   = reconstruct_video(U1_gt, U3_gt, G_gt)

            e = float(np.linalg.norm(v_gen - v_gt) /
                      (np.linalg.norm(v_gt) + 1e-8))
            rel_errs.append(e)
        m = float(np.mean(rel_errs))
        print(f'  seed {s}: rel-err {rel_errs} mean={m:.4f}')

    print('\nVerification PASSED.')


if __name__ == '__main__':
    main()
