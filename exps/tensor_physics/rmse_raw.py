"""
rmse_raw.py — Per-sample RMSE of generated videos vs RAW original videos.

For each seed, compute per-sample:
    rmse_i  = sqrt(mean((v_gen - v_raw)^2))
    rrmse_i = rmse_i / sqrt(mean(v_raw^2))

v_raw is the original simulation field (frames 1..200 of the (201,128,128) raw
trajectory). v_gen is reconstructed from the diffusion-generated Tucker factors.

Aggregate: mean over samples → per-seed; mean ± std over 5 seeds.
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


CFG = {
    'burgers': {
        'raw_test_dir' : '/anvil/projects/x-eng260004/factor_diffusion/original_data/burgers_2d/test_data',
        'raw_prefix'   : 'test_shard',
        'raw_digits'   : 5,
        'samples_per_shard': 100,
        'has_two_fields': True,    # ux (even sid), uy (odd sid)
        'recon'        : recon_burgers,
        'gen_keys'     : ('U1', 'U3', 'G'),
    },
    'karman': {
        'raw_test_dir' : '/anvil/projects/x-eng260004/factor_diffusion/original_data/karman_vortex_2d/test_data',
        'raw_prefix'   : 'test_shard',
        'raw_digits'   : 3,
        'samples_per_shard': 50,
        'has_two_fields': False,   # vor only
        'recon'        : recon_karman,
        'gen_keys'     : ('U_T', 'U_Y', 'G'),
    },
}


def raw_lookup_for_sid(cfg, sid):
    if cfg['has_two_fields']:
        phys = sid // 2
        field = 'ux' if (sid % 2 == 0) else 'uy'
    else:
        phys = sid
        field = 'vor'
    shard_idx = phys // cfg['samples_per_shard']
    local     = phys % cfg['samples_per_shard']
    path = os.path.join(
        cfg['raw_test_dir'],
        f'{cfg["raw_prefix"]}_{shard_idx:0{cfg["raw_digits"]}d}.pt',
    )
    return path, local, field


def process(exp, gen_dir, epoch, seeds=(0, 1, 2, 3, 4)):
    cfg = CFG[exp]
    recon = cfg['recon']
    keys  = cfg['gen_keys']

    # Cache raw shards so we don't reload per sample
    raw_cache = {}

    rmse_per_seed   = []
    rrmse_per_seed  = []
    rmse_sample_std = []

    for s in seeds:
        f = os.path.join(gen_dir, f'epoch{epoch:05d}_seed{s}.pt')
        d = torch.load(f, map_location='cpu', weights_only=False)
        A_gen = d[keys[0]].numpy().astype(np.float32)
        B_gen = d[keys[1]].numpy().astype(np.float32)
        G_gen = d[keys[2]].numpy().astype(np.float32)
        sid_arr = d['sample_idx'].numpy().astype(np.int64)
        N = A_gen.shape[0]

        rmses  = np.empty(N, dtype=np.float64)
        rrmses = np.empty(N, dtype=np.float64)
        t0 = time.time()
        for i in range(N):
            sid = int(sid_arr[i])
            path, local, field = raw_lookup_for_sid(cfg, sid)
            if path not in raw_cache:
                raw_cache[path] = torch.load(path, map_location='cpu',
                                              weights_only=False)
            raw_shard = raw_cache[path]
            v_raw = raw_shard[local][field].numpy().astype(np.float32)[1:]   # (200,H,W)

            v_gen = recon(A_gen[i], B_gen[i], G_gen[i]).astype(np.float32)
            diff  = v_gen - v_raw
            mse_d = float((diff * diff).mean())
            mse_g = float((v_raw * v_raw).mean())
            rmse  = float(np.sqrt(mse_d))
            rrmse = float(np.sqrt(mse_d / max(mse_g, 1e-30)))
            rmses[i]  = rmse
            rrmses[i] = rrmse

            if (i + 1) % 200 == 0 or (i + 1) == N:
                print(f'  [{exp} seed {s}] {i+1}/{N}  '
                      f'elapsed={time.time()-t0:.1f}s', flush=True)

        rmse_per_seed.append(float(rmses.mean()))
        rrmse_per_seed.append(float(rrmses.mean()))
        rmse_sample_std.append(float(rmses.std()))
        print(f'  [{exp} seed {s}]  '
              f'mean_RMSE={rmse_per_seed[-1]:.4e}  '
              f'mean_rRMSE={rrmse_per_seed[-1]:.4f}  '
              f'(sample-std rmse={rmse_sample_std[-1]:.4e})', flush=True)

    rmses  = np.array(rmse_per_seed)
    rrmses = np.array(rrmse_per_seed)
    print(f'\n===== [{exp} vs RAW]  N_samples={N}  seeds={list(seeds)} =====')
    print(f'  RMSE (absolute)         mean={rmses.mean():.4e}   std={rmses.std(ddof=1):.4e}')
    print(f'  rRMSE = RMSE/RMS(v_raw) mean={rrmses.mean():.4f}   std={rrmses.std(ddof=1):.4f}')
    return {
        'exp'              : exp,
        'epoch'            : epoch,
        'gen_dir'          : gen_dir,
        'reference'        : 'raw original video (frames 1..200)',
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
    out_path = f'{base}/rmse_raw_lr1e-4_epoch500.json'
    with open(out_path, 'w') as fp:
        json.dump(out, fp, indent=2)
    print(f'\nWrote {out_path}')
