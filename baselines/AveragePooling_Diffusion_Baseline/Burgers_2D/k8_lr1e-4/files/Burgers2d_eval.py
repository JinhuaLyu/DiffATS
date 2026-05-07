import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Burgers2d_DiT_Model import TrajDiT, GaussianDiffusion

# Metrics for [200, 2, 128, 128] tensors — 2 channels: ux, uy

def relative_l2(pred, gt):
    return (torch.norm(pred - gt) / torch.norm(gt)).item()

def relative_l1(pred, gt):
    return ((pred - gt).abs().sum() / gt.abs().sum()).item()

def avg_rmse(pred, gt):
    return torch.sqrt(((pred - gt) ** 2).mean()).item()

def psnr(pred, gt):
    data_range = gt.max() - gt.min()
    mse = ((pred - gt) ** 2).mean().item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range.item() ** 2 / mse)


def fmt(mean, std):
    if std == 0:
        return f"{mean:.4f} ± 0"
    exp     = int(np.floor(np.log10(abs(std))))
    coeff   = std / (10 ** exp)
    c_round = round(coeff, 1)
    std_str = f"{c_round}e{exp:+d}"
    return f"{mean:.4f} ± {std_str}"


# Evaluate one trajectory

@torch.no_grad()
def evaluate_sample(diffusion, ux_data, uy_data, pool_k, num_steps, orig_size, device):
    """
    ux_data, uy_data : each [201, 128, 128]
    Frame 0 = initial condition (t=0), frames 1..200 = trajectory.
    Model is single-channel: ux and uy are predicted independently,
    then stacked into [200, 2, 128, 128] for joint metric computation.
    """
    ux_data = ux_data.to(device)   # [201, 128, 128]
    uy_data = uy_data.to(device)   # [201, 128, 128]

    cond_ux = F.avg_pool2d(
        ux_data[0].unsqueeze(0).unsqueeze(0), pool_k   # [1, 1, 16, 16]
    )
    with torch.amp.autocast('cuda'):
        traj_ux_lr = diffusion.ddim_sample(cond_ux, num_steps=num_steps)  # [1, 200, 16, 16]

    
    cond_uy = F.avg_pool2d(
        uy_data[0].unsqueeze(0).unsqueeze(0), pool_k   # [1, 1, 16, 16]
    )
    with torch.amp.autocast('cuda'):
        traj_uy_lr = diffusion.ddim_sample(cond_uy, num_steps=num_steps)  # [1, 200, 16, 16]

    
    def upsample(traj_lr):
        return F.interpolate(
            traj_lr.squeeze(0).unsqueeze(1),   # [200, 1, 16, 16]
            size=(orig_size, orig_size),
            mode='bilinear', align_corners=False,
        ).squeeze(1)                            # [200, 128, 128]

    traj_ux_hr = upsample(traj_ux_lr)   # [200, 128, 128]
    traj_uy_hr = upsample(traj_uy_lr)   # [200, 128, 128]

    # stack into [200, 2, 128, 128] for joint eval
    pred = torch.stack([traj_ux_hr, traj_uy_hr], dim=1)   # [200, 2, 128, 128]
    gt   = torch.stack([ux_data[1:], uy_data[1:]], dim=1) # [200, 2, 128, 128]

    return {
        'rel_l2':   relative_l2(pred, gt),
        'rel_l1':   relative_l1(pred, gt),
        'avg_rmse': avg_rmse(pred, gt),
        'psnr':     psnr(pred, gt),
    }


# Args

def get_args():
    p = argparse.ArgumentParser(description='Burgers 2D Evaluation')

    p.add_argument('--test_dir', type=str,
        default='${DATA_ROOT}/data/burgers_2d/test_data')
    p.add_argument('--pool_k',            type=int, default=8)
    p.add_argument('--orig_size',         type=int, default=128)
    p.add_argument('--num_test_samples',  type=int, default=500)
    p.add_argument('--samples_per_shard', type=int, default=100)
    p.add_argument('--num_seeds',         type=int, default=5)

    # model
    p.add_argument('--ckpt_path',      type=str, required=True)
    p.add_argument('--hidden_dim',     type=int, default=512)
    p.add_argument('--num_layers',     type=int, default=12)
    p.add_argument('--num_heads',      type=int, default=8)
    p.add_argument('--diff_timesteps', type=int, default=1000)
    p.add_argument('--ddim_steps', type=int, default=250)
    p.add_argument('--output_dir', type=str,
        default='${DATA_ROOT}/Experiments_Output/Burgers_2D/k8_H200_fixlr')

    return p.parse_args()


# Main

def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    
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

    test_files = sorted([
        f for f in os.listdir(args.test_dir)
        if f.endswith('.pt')
    ])
    print(f"\nTest shards: {test_files}")

    all_samples = []   # list of (ux, uy) tuples
    for shard_file in test_files:
        shard = torch.load(
            os.path.join(args.test_dir, shard_file),
            map_location='cpu', weights_only=False
        )
        for sample in shard:
            all_samples.append((sample['ux'], sample['uy']))
        if len(all_samples) >= args.num_test_samples:
            break

    all_samples = all_samples[:args.num_test_samples]
    print(f"Test trajectories : {len(all_samples)}")
    print(f"Seeds             : {args.num_seeds}")
    print(f"Total runs        : {len(all_samples) * args.num_seeds}")
    print(f"(each trajectory runs ddim_sample twice: once for ux, once for uy)")

    seed_means = {k: [] for k in ['rel_l2', 'rel_l1', 'avg_rmse', 'psnr']}

    for seed_idx in range(args.num_seeds):
        seed = seed_idx * 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n--- Seed {seed_idx+1}/{args.num_seeds}  (seed={seed}) ---")

        per_sample = {k: [] for k in ['rel_l2', 'rel_l1', 'avg_rmse', 'psnr']}

        for i, (ux_data, uy_data) in enumerate(tqdm(all_samples, desc=f'Seed {seed_idx+1}')):
            m = evaluate_sample(
                diffusion=diffusion,
                ux_data=ux_data,
                uy_data=uy_data,
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
                      f"avg_rmse={np.mean(per_sample['avg_rmse']):.4f}  "
                      f"psnr={np.mean(per_sample['psnr']):.2f}dB")

        for k in seed_means:
            seed_means[k].append(np.mean(per_sample[k]))

        print(f"Seed {seed_idx+1} summary: "
              f"rel_l2={seed_means['rel_l2'][-1]:.4f}  "
              f"rel_l1={seed_means['rel_l1'][-1]:.4f}  "
              f"avg_rmse={seed_means['avg_rmse'][-1]:.4f}  "
              f"psnr={seed_means['psnr'][-1]:.2f}dB")

    # report
    LABELS = {
        'rel_l2':   'Relative L2  ',
        'rel_l1':   'Relative L1  ',
        'avg_rmse': 'Average RMSE ',
        'psnr':     'PSNR (dB)    ',
    }

    print("\n" + "=" * 60)
    print("FINAL RESULTS  (mean ± std across 5 seeds, 500 test trajectories)")
    print("=" * 60)
    for k in ['rel_l2', 'rel_l1', 'avg_rmse', 'psnr']:
        m = np.mean(seed_means[k])
        s = np.std(seed_means[k])
        print(f"{LABELS[k]} : {fmt(m, s)}")
    print("=" * 60)

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
        for k in ['rel_l2', 'rel_l1', 'avg_rmse', 'psnr']:
            m = np.mean(seed_means[k])
            s = np.std(seed_means[k])
            f.write(f"{LABELS[k]} : {fmt(m, s)}\n")

    print(f"\nSummary saved : {txt_path}")


if __name__ == '__main__':
    main()
