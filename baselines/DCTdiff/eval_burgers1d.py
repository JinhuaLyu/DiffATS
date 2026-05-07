
import os
import time
import numpy as np
import torch
from absl import app, flags
from ml_collections import config_flags
from tqdm.auto import tqdm

import sde
from datasets_burgers import Burgers1D
from libs.uvit_pde import UViTPDE
from sde_pde import CondScoreModel, euler_maruyama_cond
from DCT_utils_1d import (
    DCT2DBlocks, reverse_zigzag_order_2d, tokens_to_field,
)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Eval configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "Path to nnet_ema.pth (or nnet.pth).")
flags.DEFINE_string("output_path", None,
                    "Where to dump the eval report and full predictions tensor.")
flags.DEFINE_integer("n_eval", -1, "Number of test trajectories to eval (-1 = all).")
flags.DEFINE_integer("batch_size", 32, "Sampling batch size.")
flags.DEFINE_integer("sample_steps", 250, "NFE.")
flags.DEFINE_string("algorithm", "ode", "ode | sde")
flags.mark_flag_as_required("nnet_path")


def _decode_tokens(tokens, ds):
    """Decode tokens to field. Uses per-freq whitening if dataset has coef_std,
    otherwise falls back to scalar Y_bound."""
    return ds.decode_tokens(tokens)


def _per_sample_metrics(pred, gt, eps=1e-8):
    """pred, gt: (B, T, X) tensors on the same device.
    Returns dict of (B,) numpy arrays for L1, L2, rMSE."""
    diff = (pred - gt).flatten(1)                          # (B, T*X)
    g = gt.flatten(1)
    rel_l1 = (diff.abs().sum(dim=1)
              / g.abs().sum(dim=1).clamp(min=eps))
    rel_l2 = (diff.pow(2).sum(dim=1).sqrt()
              / g.pow(2).sum(dim=1).sqrt().clamp(min=eps))
    rmse = diff.pow(2).mean(dim=1).sqrt()                  # absolute RMSE
    return {
        'rel_l1': rel_l1.detach().cpu().numpy(),
        'rel_l2': rel_l2.detach().cpu().numpy(),
        'rmse':   rmse.detach().cpu().numpy(),
    }


def _fmt_stderr(stderr):
    """Format stderr to 1 sig fig in scientific (e.g. 1e-4, 3e-5)."""
    if stderr <= 0:
        return '0'
    exp = int(np.floor(np.log10(stderr)))
    mant = round(stderr / (10 ** exp))
    if mant == 10:                       # rounding pushed to next decade
        mant = 1
        exp += 1
    return f'{mant}e{exp}'


def _fmt(mean, stderr):
    return f'{mean:.4f} ± {_fmt_stderr(stderr)}'


def main(argv):
    config = FLAGS.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Burgers1D(**config.dataset)
    test_ds = dataset.get_split('test')
    if test_ds is None:
        raise RuntimeError('test_path is required for eval')

    nnet = UViTPDE(**config.nnet).to(device)
    state = torch.load(FLAGS.nnet_path, map_location=device, weights_only=False)
    nnet.load_state_dict(state)
    nnet.eval()
    n_params = sum(p.numel() for p in nnet.parameters())
    print(f'loaded {FLAGS.nnet_path}  ({n_params/1e6:.2f} M params)')

    n_ic = config.dataset.n_ic_tokens
    score_model = CondScoreModel(
        nnet, pred=config.pred,
        sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale),
        n_ic_tokens=n_ic,
    )

    N = len(test_ds)
    n_eval = N if FLAGS.n_eval == -1 else min(FLAGS.n_eval, N)
    print(f'evaluating {n_eval} of {N} test trajectories  '
          f'(algorithm={FLAGS.algorithm}, steps={FLAGS.sample_steps}, batch={FLAGS.batch_size})')

    test_dict = torch.load(config.dataset.test_path, map_location='cpu', weights_only=False)
    nu_idx = test_dict['nu_index'][:n_eval].cpu().numpy()  # (n_eval,)

    if FLAGS.output_path:
        os.makedirs(FLAGS.output_path, exist_ok=True)

    # Pre-allocate full generated tensor (saved to scratch at end).
    all_gen = torch.zeros(n_eval, 200, test_ds.X, dtype=torch.float32)
    all_gt = test_ds.u[:n_eval, 1:201].clone().float()

    bsz = FLAGS.batch_size
    metric_lists = {'rel_l1': [], 'rel_l2': [], 'rmse': []}
    t_start = time.time()

    for i in tqdm(range(0, n_eval, bsz), desc='eval'):
        ids = list(range(i, min(i + bsz, n_eval)))
        toks_list, nu_list = [], []
        for j in ids:
            t, nu_v, _ = test_ds[j]
            toks_list.append(t)
            nu_list.append(nu_v)
        x_target = torch.stack(toks_list).to(device)
        nu = torch.stack(nu_list).to(device)
        ic_clean = x_target[:, :n_ic]
        x_init = torch.randn_like(x_target)

        if FLAGS.algorithm == 'ode':
            rsde = sde.ODE(score_model)
        else:
            rsde = sde.ReverseSDE(score_model)
        x_gen = euler_maruyama_cond(
            rsde, x_init, sample_steps=FLAGS.sample_steps,
            n_ic_tokens=n_ic, ic_clean=ic_clean, nu=nu,
        )

        field_gen = _decode_tokens(x_gen, test_ds)         # (B, T_pad, X)
        # Slice off the actual generation rows. With the current layout these
        # start at `gen_start` (= 40 for B_t=10) and span 200 rows.
        gs = test_ds.gen_start
        gen_rows = field_gen[:, gs:gs + 200]               # (B, 200, X)
        gt_rows = test_ds.u[ids, 1:201].to(device).float() # (B, 200, X)

        m = _per_sample_metrics(gen_rows, gt_rows)
        for k, v in m.items():
            metric_lists[k].append(v)
        all_gen[ids] = gen_rows.detach().cpu()

    elapsed = time.time() - t_start
    rel_l1 = np.concatenate(metric_lists['rel_l1'])
    rel_l2 = np.concatenate(metric_lists['rel_l2'])
    rmse   = np.concatenate(metric_lists['rmse'])
    sqrtN  = np.sqrt(n_eval)

    rl1_m, rl1_s = rel_l1.mean(), rel_l1.std() / sqrtN
    rl2_m, rl2_s = rel_l2.mean(), rel_l2.std() / sqrtN
    rms_m, rms_s = rmse.mean(),   rmse.std()   / sqrtN

    print()
    print('=' * 72)
    print(f'n_eval = {n_eval}   sampling time = {elapsed:.1f} s '
          f'({elapsed/n_eval:.2f} s/traj)')
    print('-' * 72)
    print('Average Relative Error L1\tAverage Relative Error L2\tAverage rMSE')
    print(f'{_fmt(rl1_m, rl1_s)}\t{_fmt(rl2_m, rl2_s)}\t{_fmt(rms_m, rms_s)}')
    print('=' * 72)

    print('\nPer-nu_index breakdown (mean rel-L2):')
    for k in sorted(np.unique(nu_idx).tolist()):
        sel = nu_idx == k
        if sel.sum() == 0:
            continue
        print(f'  nu_idx={k}  n={int(sel.sum()):3d}  '
              f'L1={rel_l1[sel].mean():.4f}  '
              f'L2={rel_l2[sel].mean():.4f}  '
              f'rMSE={rmse[sel].mean():.4f}')

    if FLAGS.output_path:
        # Per-sample metrics + nu indices
        np.save(os.path.join(FLAGS.output_path, 'metrics.npy'), {
            'rel_l1': rel_l1,
            'rel_l2': rel_l2,
            'rmse':   rmse,
            'nu_idx': nu_idx,
            'summary': {
                'rel_l1_mean': float(rl1_m), 'rel_l1_stderr': float(rl1_s),
                'rel_l2_mean': float(rl2_m), 'rel_l2_stderr': float(rl2_s),
                'rmse_mean':   float(rms_m), 'rmse_stderr':   float(rms_s),
                'n_eval': int(n_eval),
            },
        })
        # Full predictions and ground truth (for inspection / further analysis)
        torch.save({
            'pred': all_gen.float(),                # (n_eval, 200, X)
            'gt':   all_gt.float(),
            'nu_idx': torch.from_numpy(nu_idx),
            't_coord': test_dict['t_coord'][1:201],
            'x_coord': test_dict['x_coord'],
        }, os.path.join(FLAGS.output_path, 'predictions.pt'))

        with open(os.path.join(FLAGS.output_path, 'report.txt'), 'w') as f:
            f.write('Average Relative Error L1\tAverage Relative Error L2\tAverage rMSE\n')
            f.write(f'{_fmt(rl1_m, rl1_s)}\t{_fmt(rl2_m, rl2_s)}\t{_fmt(rms_m, rms_s)}\n')

        print(f'\nsaved -> {FLAGS.output_path}/  (metrics.npy, predictions.pt, report.txt)')


if __name__ == '__main__':
    app.run(main)
