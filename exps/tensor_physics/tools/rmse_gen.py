"""
rmse_gen.py — Per-sample RMSE for generated vs GT Tucker-reconstructed videos.

For each seed, compute per-sample:
    rmse_i  = sqrt(mean((v_gen - v_gt)^2))
    rrmse_i = rmse_i / sqrt(mean(v_gt^2))   (= L2 rel-err)

Aggregate: mean over samples → per-seed score; mean ± std over 5 seeds.
Both burgers (lr=1e-4) and karman (lr=1e-4) reported.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/home/x-jlyu5/jinhua/factor_diffusion/video')


def recon_karman(A, B, G):
    temp = np.einsum('txk,wk->txw', G, B, optimize=True)
    return np.einsum('ti,ixw->txw', A, temp, optimize=True)


def recon_burgers(A, B, G):
    temp = np.einsum('thk,wk->thw', G, B, optimize=True)
    return np.einsum('ti,ihw->thw', A, temp, optimize=True)


def load_gt_factors(exp):
    if exp == 'karman':
        sys.path.insert(0, '/home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_karman_vortex/train')
        from dataset_karman_2d import KarmanTucker2DDataset as DS
        train = DS('/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/karman_vortex_2d/tucker_karman_rT10_rX128_rY30', split='all', device='cpu')
        test  = DS('/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/karman_vortex_2d/tucker_karman_rT10_rX128_rY30/test_data', split='all', device='cpu', external_stats=train.stats)
        return (test.UT_all.numpy().astype(np.float32),
                test.UY_all.numpy().astype(np.float32),
                test.G_all.numpy().astype(np.float32))
    sys.path.insert(0, '/home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_burgers_2d/train')
    from dataset_burgers_2d import BurgersTucker2DDataset as DS
    train = DS('/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/burgers_2d/tucker_burgers_rT5_rH20_rW20', split='all', device='cpu')
    test  = DS('/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/burgers_2d/tucker_burgers_rT5_rH20_rW20/test_data', split='all', device='cpu', external_stats=train.stats)
    return (test.U1_all.numpy().astype(np.float32),
            test.U3_all.numpy().astype(np.float32),
            test.G_all.numpy().astype(np.float32))


def process(exp, gen_dir, epoch, seeds=(0, 1, 2, 3, 4)):
    A_gt, B_gt, G_gt = load_gt_factors(exp)
    keys = ('U_T', 'U_Y', 'G') if exp == 'karman' else ('U1', 'U3', 'G')
    recon = recon_karman if exp == 'karman' else recon_burgers

    rmse_per_seed   = []
    rrmse_per_seed  = []
    rmse_sample_std = []

    for s in seeds:
        f = os.path.join(gen_dir, f'epoch{epoch:05d}_seed{s}.pt')
        d = torch.load(f, map_location='cpu', weights_only=False)
        A_gen = d[keys[0]].numpy().astype(np.float32)
        B_gen = d[keys[1]].numpy().astype(np.float32)
        G_gen = d[keys[2]].numpy().astype(np.float32)
        sid   = d['sample_idx'].numpy().astype(np.int64)
        N = A_gen.shape[0]

        rmses  = np.empty(N, dtype=np.float64)
        rrmses = np.empty(N, dtype=np.float64)
        t0 = time.time()
        for i in range(N):
            v_gen = recon(A_gen[i], B_gen[i], G_gen[i])
            ti    = int(sid[i])
            v_gt  = recon(A_gt[ti], B_gt[ti], G_gt[ti])
            d2    = v_gen - v_gt
            mse   = (d2 * d2).mean()
            rmse  = float(np.sqrt(mse))
            rrmse = float(np.sqrt(mse / max(((v_gt * v_gt).mean()), 1e-30)))
            rmses[i]  = rmse
            rrmses[i] = rrmse
        rmse_per_seed.append(float(rmses.mean()))
        rrmse_per_seed.append(float(rrmses.mean()))
        rmse_sample_std.append(float(rmses.std()))
        print(f'  [{exp} seed {s}]  N={N}  '
              f'mean_RMSE={rmse_per_seed[-1]:.4e}  '
              f'mean_rRMSE={rrmse_per_seed[-1]:.4f}  '
              f'(sample-std rmse={rmse_sample_std[-1]:.4e})  '
              f'elapsed={time.time()-t0:.1f}s', flush=True)

    rmses  = np.array(rmse_per_seed)
    rrmses = np.array(rrmse_per_seed)
    print(f'\n===== [{exp}]  N_samples={N}  seeds={list(seeds)} =====')
    print(f'  RMSE (absolute)         mean={rmses.mean():.4e}   std={rmses.std(ddof=1):.4e}')
    print(f'  rRMSE = RMSE/RMS(v_gt)  mean={rrmses.mean():.4f}   std={rrmses.std(ddof=1):.4f}')
    return {
        'exp'              : exp,
        'epoch'            : epoch,
        'seeds'            : list(seeds),
        'N_samples'        : N,
        'per_seed_RMSE'    : rmse_per_seed,
        'per_seed_rRMSE'   : rrmse_per_seed,
        'RMSE_mean'        : float(rmses.mean()),
        'RMSE_std'         : float(rmses.std(ddof=1)),
        'rRMSE_mean'       : float(rrmses.mean()),
        'rRMSE_std'        : float(rrmses.std(ddof=1)),
    }


if __name__ == '__main__':
    import json
    base = '/anvil/projects/x-eng260004/factor_diffusion/our_method_generation'
    cfgs = [
        ('karman',  f'{base}/karman_vortex_2d',   500),
        ('burgers', f'{base}/burgers_2d',         500),
    ]
    out = []
    for exp, d, ep in cfgs:
        out.append(process(exp, d, ep))

    with open(f'{base}/rmse_lr1e-4_epoch500.json', 'w') as fp:
        json.dump(out, fp, indent=2)
    print(f'\nWrote {base}/rmse_lr1e-4_epoch500.json')
