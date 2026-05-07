"""
metrics_raw.py — Three per-sample error metrics of generated videos vs RAW
original simulation data.

For each test sample i in each seed s:
    L1_rel  = ||v_gen - v_raw||_1 / ||v_raw||_1
    L2_rel  = ||v_gen - v_raw||_2 / ||v_raw||_2     (= rRMSE)
    RMSE    = sqrt(mean((v_gen - v_raw)^2))         (absolute)

Aggregate: mean over samples → per-seed score; mean ± std over 5 seeds.

v_raw : raw simulation field, frames 1..200 (shape (200, 128, 128))
v_gen : reconstruct(generated Tucker factors)
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '${REPO_ROOT}/video')


def recon_karman(A, B, G):
    temp = np.einsum('txk,wk->txw', G, B, optimize=True)
    return np.einsum('ti,ixw->txw', A, temp, optimize=True)


def recon_burgers(A, B, G):
    temp = np.einsum('thk,wk->thw', G, B, optimize=True)
    return np.einsum('ti,ihw->thw', A, temp, optimize=True)


CFG = {
    'burgers': {
        'raw_test_dir' : '${DATA_ROOT}/original_data/burgers_2d/test_data',
        'raw_prefix'   : 'test_shard',
        'raw_digits'   : 5,
        'samples_per_shard': 100,
        'has_two_fields': True,
        'recon'        : recon_burgers,
        'gen_keys'     : ('U1', 'U3', 'G'),
    },
    'karman': {
        'raw_test_dir' : '${DATA_ROOT}/original_data/karman_vortex_2d/test_data',
        'raw_prefix'   : 'test_shard',
        'raw_digits'   : 3,
        'samples_per_shard': 50,
        'has_two_fields': False,
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

    raw_cache = {}

    per_seed_L1   = []
    per_seed_L2   = []
    per_seed_RMSE = []

    for s in seeds:
        f = os.path.join(gen_dir, f'epoch{epoch:05d}_seed{s}.pt')
        d = torch.load(f, map_location='cpu', weights_only=False)
        A_gen = d[keys[0]].numpy().astype(np.float32)
        B_gen = d[keys[1]].numpy().astype(np.float32)
        G_gen = d[keys[2]].numpy().astype(np.float32)
        sid_arr = d['sample_idx'].numpy().astype(np.int64)
        N = A_gen.shape[0]

        l1s = np.empty(N, dtype=np.float64)
        l2s = np.empty(N, dtype=np.float64)
        rmses = np.empty(N, dtype=np.float64)
        t0 = time.time()
        for i in range(N):
            sid = int(sid_arr[i])
            path, local, field = raw_lookup_for_sid(cfg, sid)
            if path not in raw_cache:
                raw_cache[path] = torch.load(path, map_location='cpu',
                                              weights_only=False)
            v_raw = raw_cache[path][local][field].numpy().astype(np.float32)[1:]
            v_gen = recon(A_gen[i], B_gen[i], G_gen[i]).astype(np.float32)
            diff  = v_gen - v_raw

            l1_diff = float(np.abs(diff).sum())
            l1_raw  = float(np.abs(v_raw).sum())
            l2_diff = float(np.linalg.norm(diff))
            l2_raw  = float(np.linalg.norm(v_raw))
            rmse    = float(np.sqrt((diff * diff).mean()))

            l1s[i]   = l1_diff / max(l1_raw, 1e-30)
            l2s[i]   = l2_diff / max(l2_raw, 1e-30)
            rmses[i] = rmse

            if (i + 1) % 200 == 0 or (i + 1) == N:
                print(f'  [{exp} seed {s}] {i+1}/{N}  '
                      f'elapsed={time.time()-t0:.1f}s', flush=True)

        per_seed_L1.append(float(l1s.mean()))
        per_seed_L2.append(float(l2s.mean()))
        per_seed_RMSE.append(float(rmses.mean()))
        print(f'  [{exp} seed {s}]  '
              f'L1={per_seed_L1[-1]:.4f}  '
              f'L2={per_seed_L2[-1]:.4f}  '
              f'RMSE={per_seed_RMSE[-1]:.4e}', flush=True)

    L1   = np.array(per_seed_L1)
    L2   = np.array(per_seed_L2)
    RMSE = np.array(per_seed_RMSE)
    print(f'\n===== [{exp} vs RAW]  N_samples={N}  seeds={list(seeds)} =====')
    print(f'  L1 rel-err   mean={L1.mean():.4f}     std={L1.std(ddof=1):.4f}')
    print(f'  L2 rel-err   mean={L2.mean():.4f}     std={L2.std(ddof=1):.4f}')
    print(f'  RMSE (abs)   mean={RMSE.mean():.4e}   std={RMSE.std(ddof=1):.4e}')
    return {
        'exp'              : exp,
        'epoch'            : epoch,
        'gen_dir'          : gen_dir,
        'reference'        : 'raw simulation video frames 1..200',
        'seeds'            : list(seeds),
        'N_samples'        : N,
        'per_seed_L1'      : per_seed_L1,
        'per_seed_L2'      : per_seed_L2,
        'per_seed_RMSE'    : per_seed_RMSE,
        'L1_mean'          : float(L1.mean()),
        'L1_std'           : float(L1.std(ddof=1)),
        'L2_mean'          : float(L2.mean()),
        'L2_std'           : float(L2.std(ddof=1)),
        'RMSE_mean'        : float(RMSE.mean()),
        'RMSE_std'         : float(RMSE.std(ddof=1)),
    }


if __name__ == '__main__':
    import json
    base = '${DATA_ROOT}/our_method_generation'
    cfgs = [
        ('karman',  f'{base}/karman_vortex_2d',   500),
        ('burgers', f'{base}/burgers_2d',         500),
    ]
    out = []
    for exp, d, ep in cfgs:
        out.append(process(exp, d, ep))
    out_path = f'{base}/metrics_raw_lr1e-4_epoch500.json'
    with open(out_path, 'w') as fp:
        json.dump(out, fp, indent=2)
    print(f'\nWrote {out_path}')
