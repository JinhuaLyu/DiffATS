import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Burgers1d_DiT_Model import TrajDiT1D, GaussianDiffusion

# Metrics for [200, 1024] tensors

def relative_l2(pred, gt):
    return (torch.norm(pred - gt) / torch.norm(gt)).item()

def relative_l1(pred, gt):
    return ((pred - gt).abs().sum() / gt.abs().sum()).item()

def rmse(pred, gt):
    return torch.sqrt(((pred - gt) ** 2).mean()).item()

def psnr(pred, gt):
    data_range = gt.max() - gt.min()
    mse = ((pred - gt) ** 2).mean().item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range.item() ** 2 / mse)

def fmt(mean, std):
    if std == 0:
        return f"{mean:.5f} ± 0"
    exp     = int(np.floor(np.log10(abs(std))))
    coeff   = std / (10 ** exp)
    c_round = round(coeff, 1)
    std_str = f"{c_round}e{exp:+d}"
    return f"{mean:.5f} ± {std_str}"


# Evaluate one trajectory

@torch.no_grad()
def evaluate_sample(diffusion, cond_spatial, cond_nu, gt_hr, pool_k,
                    num_steps, orig_L, device):
    with torch.amp.autocast('cuda'):
        pred_lr = diffusion.ddim_sample(
            cond_spatial, cond_nu, num_steps=num_steps
        )  # [1, 200, L_lr]

    pred_hr = F.interpolate(
        pred_lr, size=orig_L, mode='linear', align_corners=False
    ).squeeze(0)  # [200, 1024]

    return {
        'rel_l2': relative_l2(pred_hr, gt_hr),
        'rel_l1': relative_l1(pred_hr, gt_hr),
        'rmse':   rmse(pred_hr, gt_hr),
        'psnr':   psnr(pred_hr, gt_hr),
    }


# Args

def get_args():
    p = argparse.ArgumentParser(description='1D Burgers Evaluation')

    p.add_argument('--test_path', type=str,
        default='/scratch/bgxp/ezhou1/factor_diffusion_proj/data/burgers_1d/burgers_1d_test.pt')
    p.add_argument('--pool_k',           type=int, default=4)
    p.add_argument('--orig_L',           type=int, default=1024)
    p.add_argument('--num_test_samples', type=int, default=500)
    p.add_argument('--num_seeds',        type=int, default=5)

    # model
    p.add_argument('--ckpt_path',      type=str, required=True)
    p.add_argument('--hidden_dim',     type=int, default=512)
    p.add_argument('--num_layers',     type=int, default=12)
    p.add_argument('--num_heads',      type=int, default=8)
    p.add_argument('--diff_timesteps', type=int, default=1000)

    # sampling
    p.add_argument('--ddim_steps', type=int, default=250)

    p.add_argument('--output_dir', type=str,
        default='/scratch/bgxp/ezhou1/factor_diffusion_proj/Experiments_Output/Burgers_1D/k4_fixlr')

    return p.parse_args()


# Main

def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    L_lr = args.orig_L // args.pool_k   # 256 for pool_k=4

    model = TrajDiT1D(
        L_lr=L_lr,
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
    print(f"\nLoading test data: {args.test_path}")
    data = torch.load(args.test_path, map_location='cpu', weights_only=False)

    trajectories = data['tensor'][:args.num_test_samples]  # [500, 201, 1024]
    nu_all       = data['nu'][:args.num_test_samples]      # [500]

    # pool condition (t=0)
    cond_spatial_all = F.avg_pool1d(
        trajectories[:, 0:1, :], kernel_size=args.pool_k
    )  # [500, 1, L_lr]

    # GT at original resolution: t=1..200
    gt_hr_all = trajectories[:, 1:, :]   # [500, 200, 1024]

    print(f"Test trajectories : {len(trajectories)}")
    print(f"Seeds             : {args.num_seeds}")

    METRICS = ['rel_l2', 'rel_l1', 'rmse', 'psnr']
    seed_means = {k: [] for k in METRICS}

    for seed_idx in range(args.num_seeds):
        seed = seed_idx * 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n--- Seed {seed_idx+1}/{args.num_seeds}  (seed={seed}) ---")

        per_sample = {k: [] for k in METRICS}

        for i in tqdm(range(len(trajectories)), desc=f'Seed {seed_idx+1}'):
            cond_sp = cond_spatial_all[i:i+1].to(device)    # [1, 1, L_lr]
            cond_nu = nu_all[i:i+1].unsqueeze(1).to(device)  # [1, 1]
            gt_hr   = gt_hr_all[i].to(device)                # [200, 1024]

            m = evaluate_sample(
                diffusion, cond_sp, cond_nu, gt_hr,
                pool_k=args.pool_k,
                num_steps=args.ddim_steps,
                orig_L=args.orig_L,
                device=device,
            )
            for k in METRICS:
                per_sample[k].append(m[k])

            if (i + 1) % 100 == 0:
                print(f"  [{i+1:4d}/500]  "
                      f"rel_l2={np.mean(per_sample['rel_l2']):.5f}  "
                      f"rel_l1={np.mean(per_sample['rel_l1']):.5f}  "
                      f"rmse={np.mean(per_sample['rmse']):.5f}  "
                      f"psnr={np.mean(per_sample['psnr']):.2f}dB")

        for k in METRICS:
            seed_means[k].append(np.mean(per_sample[k]))

        print(f"Seed {seed_idx+1}: "
              f"rel_l2={seed_means['rel_l2'][-1]:.5f}  "
              f"rel_l1={seed_means['rel_l1'][-1]:.5f}  "
              f"rmse={seed_means['rmse'][-1]:.5f}  "
              f"psnr={seed_means['psnr'][-1]:.2f}dB")
        
    LABELS = {
        'rel_l2': 'Relative L2  ',
        'rel_l1': 'Relative L1  ',
        'rmse':   'RMSE         ',
        'psnr':   'PSNR (dB)    ',
    }

    print("\n" + "=" * 60)
    print("FINAL RESULTS  (mean ± std across 5 seeds, 500 test trajectories)")
    print("=" * 60)
    for k in METRICS:
        m = np.mean(seed_means[k])
        s = np.std(seed_means[k])
        print(f"{LABELS[k]} : {fmt(m, s)}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"k{args.pool_k}_ddim{args.ddim_steps}_n{args.num_test_samples}_s{args.num_seeds}"

    np.save(os.path.join(args.output_dir, f'eval_results_{tag}.npy'), seed_means)

    txt_path = os.path.join(args.output_dir, f'eval_summary_{tag}.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Checkpoint       : {args.ckpt_path}\n")
        f.write(f"Pool k           : {args.pool_k}\n")
        f.write(f"Compression      : {args.pool_k}x\n")
        f.write(f"DDIM steps       : {args.ddim_steps}\n")
        f.write(f"Num trajectories : {len(trajectories)}\n")
        f.write(f"Num seeds        : {args.num_seeds}\n\n")
        for k in METRICS:
            m = np.mean(seed_means[k])
            s = np.std(seed_means[k])
            f.write(f"{LABELS[k]} : {fmt(m, s)}\n")

    print(f"\nSummary saved : {txt_path}")


if __name__ == '__main__':
    main()
