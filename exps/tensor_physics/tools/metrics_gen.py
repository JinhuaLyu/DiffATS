"""
metrics_gen.py — Compute per-sample relative errors (L1 and L2/Frobenius) of
generated videos vs ground-truth Tucker-reconstructed videos.

For each experiment (karman, burgers):
  - Load 5 seed files (each with denormalized Tucker factors for all test rows).
  - For each seed and each sample i:
      v_gen = reconstruct_video(factors_gen_i)
      v_gt  = reconstruct_video(test_dataset.factors[sample_idx_i])
      rel_l1 = ||v_gen - v_gt||_1 / ||v_gt||_1
      rel_l2 = ||v_gen - v_gt||_2 / ||v_gt||_2
  - Aggregate: mean over samples → per-seed score.
  - Across 5 seeds: mean ± std.

Ground truth = the Tucker-reconstruction of the stored factors, matching the
metric used in training validation.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '${REPO_ROOT}/video')


def recon_karman(A, B, G):
    # A:(T,rT)  B:(W,rY)  G:(rT,X,rY)  → (T,X,W)
    temp = np.einsum('txk,wk->txw', G, B, optimize=True)
    return np.einsum('ti,ixw->txw', A, temp, optimize=True)


def recon_burgers(A, B, G):
    # A:(T,rT)  B:(W,rW)  G:(rT,H,rW)  → (T,H,W)
    temp = np.einsum('thk,wk->thw', G, B, optimize=True)
    return np.einsum('ti,ihw->thw', A, temp, optimize=True)


def rel_err_pair(v_gen, v_gt):
    diff = v_gen - v_gt
    gt_l1 = np.abs(v_gt).sum()
    gt_l2 = np.linalg.norm(v_gt)
    l1 = float(np.abs(diff).sum() / (gt_l1 + 1e-12))
    l2 = float(np.linalg.norm(diff) / (gt_l2 + 1e-12))
    return l1, l2


def load_gt_factors(exp):
    if exp == 'karman':
        from dataset_karman_2d import KarmanTucker2DDataset
        train = KarmanTucker2DDataset(
            '${DATA_ROOT}/tucker_factors/'
            'karman_vortex_2d/tucker_karman_rT10_rX128_rY30',
            split='all', device='cpu',
        )
        test = KarmanTucker2DDataset(
            '${DATA_ROOT}/tucker_factors/'
            'karman_vortex_2d/tucker_karman_rT10_rX128_rY30/test_data',
            split='all', device='cpu',
            external_stats=train.stats,
        )
        A = test.UT_all.numpy().astype(np.float32)
        B = test.UY_all.numpy().astype(np.float32)
        G = test.G_all.numpy().astype(np.float32)
    else:
        from dataset_burgers_2d import BurgersTucker2DDataset
        train = BurgersTucker2DDataset(
            '${DATA_ROOT}/tucker_factors/'
            'burgers_2d/tucker_burgers_rT5_rH20_rW20',
            split='all', device='cpu',
        )
        test = BurgersTucker2DDataset(
            '${DATA_ROOT}/tucker_factors/'
            'burgers_2d/tucker_burgers_rT5_rH20_rW20/test_data',
            split='all', device='cpu',
            external_stats=train.stats,
        )
        A = test.U1_all.numpy().astype(np.float32)
        B = test.U3_all.numpy().astype(np.float32)
        G = test.G_all.numpy().astype(np.float32)
    return A, B, G


def process(exp, seed_files):
    if exp == 'karman':
        sys.path.insert(0, '${REPO_ROOT}/tensor_physics/exp_karman_vortex/train')
        keys = ('U_T', 'U_Y', 'G')
        recon = recon_karman
    else:
        sys.path.insert(0, '${REPO_ROOT}/tensor_physics/exp_burgers_2d/train')
        keys = ('U1', 'U3', 'G')
        recon = recon_burgers

    A_gt, B_gt, G_gt = load_gt_factors(exp)

    per_seed_l1, per_seed_l2 = [], []
    per_seed_l1_std, per_seed_l2_std = [], []
    seeds = []

    for sf in seed_files:
        d = torch.load(sf, map_location='cpu', weights_only=False)
        s = d.get('seed', -1)
        seeds.append(s)
        A_gen = d[keys[0]].numpy().astype(np.float32)
        B_gen = d[keys[1]].numpy().astype(np.float32)
        G_gen = d[keys[2]].numpy().astype(np.float32)
        sid   = d['sample_idx'].numpy().astype(np.int64)
        N = A_gen.shape[0]

        l1s = np.empty(N, dtype=np.float64)
        l2s = np.empty(N, dtype=np.float64)
        t0 = time.time()
        for i in range(N):
            v_gen = recon(A_gen[i], B_gen[i], G_gen[i])
            ti    = int(sid[i])
            v_gt  = recon(A_gt[ti], B_gt[ti], G_gt[ti])
            l1s[i], l2s[i] = rel_err_pair(v_gen, v_gt)
            if (i + 1) % 200 == 0 or (i + 1) == N:
                print(f'  [{exp} seed {s}] {i+1}/{N}  elapsed={time.time()-t0:.1f}s',
                      flush=True)

        per_seed_l1.append(float(l1s.mean()))
        per_seed_l2.append(float(l2s.mean()))
        per_seed_l1_std.append(float(l1s.std()))
        per_seed_l2_std.append(float(l2s.std()))
        print(f'  [{exp} seed {s}]  mean_L1={per_seed_l1[-1]:.4f}  '
              f'mean_L2={per_seed_l2[-1]:.4f}  '
              f'(sample-wise std: L1={per_seed_l1_std[-1]:.4f}, '
              f'L2={per_seed_l2_std[-1]:.4f})', flush=True)

    p1 = np.array(per_seed_l1); p2 = np.array(per_seed_l2)
    print(f'\n===== [{exp}] N_samples={A_gen.shape[0]}  N_seeds={len(seeds)} =====')
    print(f'  L1 rel-err  per-seed:  {[f"{x:.4f}" for x in p1]}')
    print(f'  L1 rel-err  mean={p1.mean():.4f}   std={p1.std(ddof=1):.4f}')
    print(f'  L2 rel-err  per-seed:  {[f"{x:.4f}" for x in p2]}')
    print(f'  L2 rel-err  mean={p2.mean():.4f}   std={p2.std(ddof=1):.4f}')
    return {
        'exp'          : exp,
        'seeds'        : seeds,
        'N_samples'    : int(A_gen.shape[0]),
        'per_seed_L1'  : p1.tolist(),
        'per_seed_L2'  : p2.tolist(),
        'L1_mean'      : float(p1.mean()),
        'L1_std'       : float(p1.std(ddof=1)),
        'L2_mean'      : float(p2.mean()),
        'L2_std'       : float(p2.std(ddof=1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['karman', 'burgers', 'both'],
                        default='both')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--out_json', type=str, default=None)
    args = parser.parse_args()

    base = '${DATA_ROOT}/our_method_generation'
    tag  = f'epoch{args.epoch:05d}'
    dirs = {
        'karman' : f'{base}/karman_vortex_2d',
        'burgers': f'{base}/burgers_2d',
    }

    exps = ['karman', 'burgers'] if args.exp == 'both' else [args.exp]
    results = {}
    for exp in exps:
        files = [os.path.join(dirs[exp], f'{tag}_seed{s}.pt') for s in args.seeds]
        for f in files:
            assert os.path.exists(f), f'Missing: {f}'
        results[exp] = process(exp, files)

    if args.out_json:
        import json
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nWrote {args.out_json}', flush=True)


if __name__ == '__main__':
    main()
