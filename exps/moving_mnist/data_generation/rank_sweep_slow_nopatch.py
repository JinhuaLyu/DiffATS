"""
rank_sweep_slow_nopatch.py -- Tucker rank sweep on raw (T, H, W) without patchification.

Loads slow_moving_mnist.pt, applies Tucker decomposition directly on the
(T=20, H=64, W=64) video tensor (no patch splitting), reports PSNR / SSIM /
Relative Error / Compression Ratio for each [r_T, r_H, r_W] rank combo.

Usage:
    cd ${REPO_ROOT}/exps/moving_mnist/data_generate
    python rank_sweep_slow_nopatch.py
    python rank_sweep_slow_nopatch.py \\
        --n_videos 10 --r_T_list 5 10 20 --r_H_list 16 32 --r_W_list 16 32
"""

import argparse
import os
import time

import numpy as np
import torch

_DIR         = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_DIR, 'slow_moving_mnist.pt')

DEFAULT_R_T = [4, 8, 12, 20]
DEFAULT_R_N = [16, 32, 64]
DEFAULT_R_D = [16, 32, 64]


# ---------------------------------------------------------------------------
# Tucker decomposition
# ---------------------------------------------------------------------------

def tucker_decompose(tensor_f64: np.ndarray, rank: list):
    """
    tensor_f64 : (T, H, W) float64
    rank        : [r_T, r_H, r_W]
    Returns     : core (r_T, r_H, r_W), factors, recon (T, H, W)
    """
    import tensorly as tl
    from tensorly.decomposition import tucker
    core, factors = tucker(tl.tensor(tensor_f64), rank=rank,
                           n_iter_max=100, verbose=False)
    recon = np.array(tl.tucker_to_tensor((core, factors)))
    return np.array(core), [np.array(f) for f in factors], recon


def tucker_bytes(core, factors):
    return core.nbytes + sum(f.nbytes for f in factors)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(orig, recon, max_val=255.0):
    mse = float(np.mean((orig - recon) ** 2))
    return 10.0 * np.log10(max_val ** 2 / mse) if mse > 0 else float('inf')


def _ssim_frame(a, b, data_range=255.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.std(), b.std()
    sig_ab = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a ** 2 + sig_b ** 2 + C2)
    return float(num / den)


def compute_ssim(orig, recon):
    return float(np.mean([_ssim_frame(orig[t], recon[t]) for t in range(orig.shape[0])]))


def compute_rel_error(orig, recon):
    """||orig - recon||_F / ||orig||_F"""
    norm_orig = float(np.linalg.norm(orig))
    return float(np.linalg.norm(orig - recon)) / norm_orig if norm_orig > 0 else 0.0


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_comparison_grid(originals, reconstructions, label, out_dir,
                         show_frames=(0, 4, 9, 14, 19)):
    from PIL import Image
    n_frames = len(show_frames)
    n = len(originals)
    H, W = originals[0].shape[1], originals[0].shape[2]
    canvas = np.zeros((2 * n_frames * H, n * W), dtype=np.uint8)
    for i, (orig, recon) in enumerate(zip(originals, reconstructions)):
        orig_u8  = np.clip(orig,  0, 255).astype(np.uint8)
        recon_u8 = np.clip(recon, 0, 255).astype(np.uint8)
        x = i * W
        for row, t in enumerate(show_frames):
            canvas[row * H:(row + 1) * H, x:x + W]                         = orig_u8[t]
            canvas[(n_frames + row) * H:(n_frames + row + 1) * H, x:x + W] = recon_u8[t]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'recon_nopatch_{label}.png')
    Image.fromarray(canvas, mode='L').save(path)
    print(f"  Saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   type=str, default=DATASET_PATH)
    parser.add_argument('--n_videos',  type=int, default=100)
    parser.add_argument('--start_idx', type=int, default=0)

    parser.add_argument('--ranks', type=str, default=None,
                        help='Single rank "r_T,r_H,r_W", e.g. "8,32,32"')
    parser.add_argument('--r_T_list', type=int, nargs='+', default=DEFAULT_R_T)
    parser.add_argument('--r_N_list', type=int, nargs='+', default=DEFAULT_R_N)
    parser.add_argument('--r_d_list', type=int, nargs='+', default=DEFAULT_R_D)

    parser.add_argument('--save_vis', action='store_true')
    parser.add_argument('--vis_n',    type=int, default=6)
    parser.add_argument('--out_dir',  type=str,
                        default=os.path.join(_DIR, 'rank_sweep_nopatch_output'))
    args = parser.parse_args()

    if args.ranks is not None:
        rank_candidates = [[int(x) for x in args.ranks.split(',')]]
    else:
        rank_candidates = []
        for r_T in args.r_T_list:
            for r_N in args.r_N_list:
                for r_d in args.r_d_list:
                    rank_candidates.append([r_T, r_N, r_d])

    print(f"Loading {args.dataset} ...")
    raw  = torch.load(args.dataset, weights_only=True)   # (20, N, 64, 64) uint8
    data = raw.permute(1, 0, 2, 3).numpy()               # (N, 20, 64, 64) uint8

    n      = min(args.n_videos, len(data) - args.start_idx)
    videos = data[args.start_idx: args.start_idx + n].astype(np.float64)
    T, H, W = videos.shape[1], videos.shape[2], videos.shape[3]
    orig_bytes_per_video = videos[0].nbytes

    print(f"Evaluating {n} videos  |  T={T}  H={H}  W={W}  (no patchification)")
    print(f"Ranks to sweep: {rank_candidates}\n")

    results = []
    for rank in rank_candidates:
        r_T, r_N, r_d = rank
        rank_cap = [min(r_T, T), min(r_N, H), min(r_d, W)]
        label = f"rT{rank_cap[0]}_rN{rank_cap[1]}_rd{rank_cap[2]}"

        psnr_list, ssim_list, rel_err_list = [], [], []
        vis_origs, vis_recons = [], []
        t0 = time.time()

        for i in range(n):
            core, factors, recon = tucker_decompose(videos[i], rank_cap)
            recon = np.clip(recon, 0, 255)

            psnr_list.append(compute_psnr(videos[i], recon))
            ssim_list.append(compute_ssim(videos[i], recon))
            rel_err_list.append(compute_rel_error(videos[i], recon))

            if args.save_vis and i < args.vis_n:
                vis_origs.append(videos[i])
                vis_recons.append(recon)

        elapsed      = time.time() - t0
        tb           = tucker_bytes(core, factors)
        cr           = orig_bytes_per_video / tb
        mean_psnr    = float(np.mean(psnr_list))
        mean_ssim    = float(np.mean(ssim_list))
        mean_rel_err = float(np.mean(rel_err_list))

        results.append(dict(rank=rank_cap, label=label, cr=cr,
                            psnr=mean_psnr, ssim=mean_ssim,
                            rel_err=mean_rel_err, elapsed=elapsed))

        print(f"  [{label}]  CR={cr:.1f}x  PSNR={mean_psnr:.2f}dB  "
              f"SSIM={mean_ssim:.4f}  RelErr={mean_rel_err:.4f}  ({elapsed:.1f}s)")

        if args.save_vis:
            save_comparison_grid(vis_origs, vis_recons, label, args.out_dir)

    # Summary table
    print()
    print('=' * 85)
    print('  Tucker rank sweep -- slow Moving MNIST  [NO patchification]  (T, N, d) = (T, H, W) direct')
    print('=' * 85)
    hdr = (f"{'Rank [T,N,d]':>22}  {'CR':>6}  {'PSNR (dB)':>10}  "
           f"{'SSIM':>8}  {'Rel.Err':>8}  {'time(s)':>8}")
    print(hdr)
    print('-' * len(hdr))
    for res in results:
        print(f"  {str(res['rank']):>20}  {res['cr']:>6.1f}x  "
              f"{res['psnr']:>10.2f}  {res['ssim']:>8.4f}  "
              f"{res['rel_err']:>8.4f}  {res['elapsed']:>8.1f}")
    print()

    good = [r for r in results if r['ssim'] >= 0.92]
    if good:
        best = max(good, key=lambda r: r['cr'])
        print(f"* Recommended (SSIM>=0.92, max CR): {best['rank']}  "
              f"CR={best['cr']:.1f}x  PSNR={best['psnr']:.2f}dB  "
              f"SSIM={best['ssim']:.4f}  RelErr={best['rel_err']:.4f}")
    elif results:
        best = max(results, key=lambda r: r['psnr'])
        print(f"* Best PSNR (no rank met SSIM>=0.92): {best['rank']}  "
              f"PSNR={best['psnr']:.2f}dB  RelErr={best['rel_err']:.4f}")


if __name__ == '__main__':
    main()