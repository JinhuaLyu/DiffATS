import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Karman2d_DiT_Model import TrajDiT, GaussianDiffusion


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_path', type=str,
        default='${DATA_ROOT}/pde_samples/2d_karman/pde_samples_10.pt')
    p.add_argument('--ckpt_path', type=str,
        default='${DATA_ROOT}/baseline_checkpoint/average_pooling/Karman2d_k8_latest.pt')
    p.add_argument('--output_path', type=str,
        default='${DATA_ROOT}/pde_samples_generated/2dkarman/AveragePooling/avgpool_karman2d_generated.pt')
    p.add_argument('--pool_k',    type=int, default=8)
    p.add_argument('--orig_size', type=int, default=128)
    p.add_argument('--hidden_dim',     type=int, default=512)
    p.add_argument('--num_layers',     type=int, default=12)
    p.add_argument('--num_heads',      type=int, default=8)
    p.add_argument('--diff_timesteps', type=int, default=1000)
    p.add_argument('--ddim_steps',     type=int, default=250)
    p.add_argument('--seed',           type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    spatial = args.orig_size // args.pool_k

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
    print(f"Loaded ckpt: {args.ckpt_path}  epoch={ckpt.get('epoch')}")

    data = torch.load(args.test_path, map_location='cpu', weights_only=False)
    trajectories = data['tensor']        # [10, 201, 128, 128] (vorticity)
    N = trajectories.shape[0]
    print(f"Num test trajectories: {N}")

    # SDIFT-style output: [N, 200, 1, 128, 128] (no t=0)
    generated    = torch.zeros(N, 200, 1, args.orig_size, args.orig_size, dtype=torch.float32)
    ground_truth = torch.zeros(N, 200, 1, args.orig_size, args.orig_size, dtype=torch.float32)

    for i in tqdm(range(N), desc='sampling'):
        field = trajectories[i].to(device)             # [201, 128, 128]
        cond_lr = F.avg_pool2d(
            field[0].unsqueeze(0).unsqueeze(0), args.pool_k
        )                                              # [1, 1, 16, 16]
        with torch.no_grad(), torch.amp.autocast('cuda'):
            traj_lr = diffusion.ddim_sample(cond_lr, num_steps=args.ddim_steps)
        traj_hr = F.interpolate(
            traj_lr.squeeze(0).unsqueeze(1),
            size=(args.orig_size, args.orig_size),
            mode='bilinear', align_corners=False,
        )                                              # [200, 1, 128, 128]
        generated[i]    = traj_hr.cpu().float()
        ground_truth[i] = field[1:].unsqueeze(1).cpu().float()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out = {
        'generated':    generated,
        'ground_truth': ground_truth,
        'n_init_frames': 1,
        'niu':           data.get('niu'),
        'Re':            data.get('Re'),
        'cx':            data.get('cx'),
        'cy':            data.get('cy'),
        'r':             data.get('r'),
        'param_idx':     data.get('param_idx'),
        'clip_idx':      data.get('clip_idx'),
        'step_start':    data.get('step_start'),
        'model':         'AveragePooling_DiT (k=8)',
        'checkpoint':    args.ckpt_path,
        'ddim_steps':    args.ddim_steps,
        'seed':          args.seed,
    }
    torch.save(out, args.output_path)
    print(f"Saved: {args.output_path}")
    print(f"generated shape: {generated.shape}")


if __name__ == '__main__':
    main()
