# Evaluate trajectory-level diffusion for Karman Vortex 2D
# Metrics: relative L1 / L2 / PSNR / relative RMSE
# 500 test trajectories, 5 random seeds (different initial noise)
# Report: mean ± std across 5 seeds, format: 0.0517 ± 3e-5

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Traj_dit_model import TrajDiT, GaussianDiffusion

# Metrics  (all operate on [200, 128, 128] tensors)

def relative_l2(pred, gt):
    return (torch.norm(pred - gt) / torch.norm(gt)).item()

def relative_l1(pred, gt):
    return ((pred - gt).abs().sum() / gt.abs().sum()).item()

def relative_rmse(pred, gt):
    return (torch.sqrt(((pred - gt) ** 2).mean()) /
            torch.sqrt((gt ** 2).mean())).item()

def psnr(pred, gt):
    data_range = gt.max() - gt.min()
    mse = ((pred - gt) ** 2).mean().item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range.item() ** 2 / mse)

# Format: 0.0517 ± 3e-5

def fmt(mean, std):
    if std == 0:
        return f"{mean:.4f} ± 0"
    exp     = int(np.floor(np.log10(abs(std))))
    coeff   = std / (10 ** exp)
    c_round = round(coeff, 1)
    std_str = f"{c_round}e{exp:+d}"
    return f"{mean:.4f} ± {std_str}"


# Evaluate one trajectory (one seed's noise)

@torch.no_grad()
def evaluate_sample(diffusion, field_data, pool_k, num_steps, orig_size, device):
    """
    field_data : [201, 128, 128]
    returns    : dict of 4 metrics
    """
    field_data   = field_data.to(device)
    condition_hr = field_data[0]    # [128, 128]
    gt_hr        = field_data[1:]   # [200, 128, 128]

    cond_lr = F.avg_pool2d(
        condition_hr.unsqueeze(0).unsqueeze(0), pool_k
    )  # [1, 1, 16, 16]

    with torch.amp.autocast('cuda'):
        traj_lr = diffusion.ddim_sample(cond_lr, num_steps=num_steps)
    # traj_lr: [1, 200, 16, 16]

    traj_hr = F.interpolate(
        traj_lr.squeeze(0).unsqueeze(1),   # [200, 1, 16, 16]
        size=(orig_size, orig_size),
        mode='bilinear', align_corners=False,
    ).squeeze(1)                           # [200, 128, 128]

    return {
        'rel_l2':   relative_l2(traj_hr, gt_hr),
        'rel_l1':   relative_l1(traj_hr, gt_hr),
        'rel_rmse': relative_rmse(traj_hr, gt_hr),
        'psnr':     psnr(traj_hr, gt_hr),
    }


# Args

def get_args():
    p = argparse.ArgumentParser(description='Karman Vortex Evaluation')

    p.add_argument('--test_dir', type=str,
        default='/scratch/bgxp/ezhou1/factor_diffusion_proj/data/karman_vortex_2d/test_data')
    p.add_argument('--pool_k',            type=int, default=8)
    p.add_argument('--orig_size',         type=int, default=128)
    p.add_argument('--num_test_samples',  type=int, default=500)
    p.add_argument('--samples_per_shard', type=int, default=50)
    p.add_argument('--num_seeds',         type=int, default=5)

    # model
    p.add_argument('--ckpt_path',      type=str, required=True)
    p.add_argument('--hidden_dim',     type=int, default=512)
    p.add_argument('--num_layers',     type=int, default=12)
    p.add_argument('--num_heads',      type=int, default=8)
    p.add_argument('--diff_timesteps', type=int, default=1000)

    # sampling
    p.add_argument('--ddim_steps', type=int, default=250)

    # output
    p.add_argument('--output_dir', type=str,
        default='/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/Karman_Vortex_2D/k8_H200_fixlr')

    return p.parse_args()


# Main

def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- load model ----------------------------------------------- #
    spatial = args.orig_size // args.pool_k   # 16

    model = TrajDiT(
        spatial_size=spatial,
        num_frames=200,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)

    diffusion = GaussianDiffusion(model, timesteps=args.diff_timesteps).to(device)

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded  : {args.ckpt_path}")
    print(f"  epoch={ckpt['epoch']}")

    # ---- load test data ------------------------------------------- #
    test_files = sorted([
        f for f in os.listdir(args.test_dir)
        if f.endswith('.pt')
    ])
    print(f"\nTest shards: {test_files}")

    all_samples = []
    for shard_file in test_files:
        shard = torch.load(
            os.path.join(args.test_dir, shard_file),
            map_location='cpu', weights_only=False
        )
        for sample in shard:
            all_samples.append(sample['vor'])
        if len(all_samples) >= args.num_test_samples:
            break

    all_samples = all_samples[:args.num_test_samples]
    print(f"Test trajectories : {len(all_samples)}")
    print(f"Seeds             : {args.num_seeds}")
    print(f"Total runs        : {len(all_samples) * args.num_seeds}")

    # ---- 5 seeds x 500 samples ------------------------------------ #
    # seed_means[metric] = list of per-seed mean (length = num_seeds)
    seed_means = {k: [] for k in ['rel_l2', 'rel_l1', 'rel_rmse', 'psnr']}

    for seed_idx in range(args.num_seeds):
        seed = seed_idx * 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n--- Seed {seed_idx+1}/{args.num_seeds}  (seed={seed}) ---")

        per_sample = {k: [] for k in ['rel_l2', 'rel_l1', 'rel_rmse', 'psnr']}

        for i, field_data in enumerate(tqdm(all_samples, desc=f'Seed {seed_idx+1}')):
            m = evaluate_sample(
                diffusion=diffusion,
                field_data=field_data,
                pool_k=args.pool_k,
                num_steps=args.ddim_steps,
                orig_size=args.orig_size,
                device=device,
            )
            for k in per_sample:
                per_sample[k].append(m[k])

            if (i + 1) % 100 == 0:
                print(f"  [{i+1:4d}/{len(all_samples)}]  "
                      f"rel_l2={np.mean(per_sample['rel_l2']):.4f}  "
                      f"rel_l1={np.mean(per_sample['rel_l1']):.4f}  "
                      f"rel_rmse={np.mean(per_sample['rel_rmse']):.4f}  "
                      f"psnr={np.mean(per_sample['psnr']):.2f}dB")

        # mean across 500 samples for this seed
        for k in seed_means:
            seed_means[k].append(np.mean(per_sample[k]))

        print(f"Seed {seed_idx+1} summary: "
              f"rel_l2={seed_means['rel_l2'][-1]:.4f}  "
              f"rel_l1={seed_means['rel_l1'][-1]:.4f}  "
              f"rel_rmse={seed_means['rel_rmse'][-1]:.4f}  "
              f"psnr={seed_means['psnr'][-1]:.2f}dB")

    # ---- report: mean ± std across 5 seeds ----------------------- #
    LABELS = {
        'rel_l2':   'Relative L2  ',
        'rel_l1':   'Relative L1  ',
        'rel_rmse': 'Relative RMSE',
        'psnr':     'PSNR (dB)    ',
    }

    print("\n" + "=" * 60)
    print("FINAL RESULTS  (mean ± std across 5 seeds, 500 test trajectories)")
    print("=" * 60)
    for k in ['rel_l2', 'rel_l1', 'rel_rmse', 'psnr']:
        m = np.mean(seed_means[k])
        s = np.std(seed_means[k])
        print(f"{LABELS[k]} : {fmt(m, s)}")
    print("=" * 60)

    # ---- save ----------------------------------------------------- #
    os.makedirs(args.output_dir, exist_ok=True)
    tag = (f"k{args.pool_k}_ddim{args.ddim_steps}"
           f"_n{args.num_test_samples}_s{args.num_seeds}")

    np.save(os.path.join(args.output_dir, f'eval_results_{tag}.npy'), seed_means)

    txt_path = os.path.join(args.output_dir, f'eval_summary_{tag}.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Checkpoint       : {args.ckpt_path}\n")
        f.write(f"Pool k           : {args.pool_k}\n")
        f.write(f"Compression      : {args.pool_k**2}x\n")
        f.write(f"DDIM steps       : {args.ddim_steps}\n")
        f.write(f"Num trajectories : {len(all_samples)}\n")
        f.write(f"Num seeds        : {args.num_seeds}\n\n")
        for k in ['rel_l2', 'rel_l1', 'rel_rmse', 'psnr']:
            m = np.mean(seed_means[k])
            s = np.std(seed_means[k])
            f.write(f"{LABELS[k]} : {fmt(m, s)}\n")

    print(f"\nSummary saved : {txt_path}")


if __name__ == '__main__':
    main()