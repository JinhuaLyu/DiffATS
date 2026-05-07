import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Burgers1d_DiT_Model import TrajDiT1D, GaussianDiffusion


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_path', type=str,
        default='${DATA_ROOT}/pde_samples/1d_burgers/pde_samples_10.pt')
    p.add_argument('--ckpt_path', type=str,
        default='${DATA_ROOT}/baseline_checkpoint/average_pooling/Burgers1d_k4_latest.pt')
    p.add_argument('--output_path', type=str,
        default='${DATA_ROOT}/pde_samples_generated/1d_burgers/AveragePooling/avgpool_burgers1d_generated.pt')
    p.add_argument('--pool_k', type=int, default=4)
    p.add_argument('--orig_L', type=int, default=1024)
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

    L_lr = args.orig_L // args.pool_k

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
    print(f"Loaded ckpt: {args.ckpt_path}  epoch={ckpt.get('epoch')}")

    data = torch.load(args.test_path, map_location='cpu', weights_only=False)
    trajectories = data['tensor']     # [10, 201, 1024]
    nu_all       = data['nu']         # [10]
    N = trajectories.shape[0]
    print(f"Num test trajectories: {N}")

    cond_spatial_all = F.avg_pool1d(
        trajectories[:, 0:1, :], kernel_size=args.pool_k
    )  # [N, 1, L_lr]

    preds = torch.zeros(N, 201, args.orig_L, dtype=torch.float32)
    preds[:, 0, :] = trajectories[:, 0, :]   # t=0 IC = ground truth

    for i in tqdm(range(N), desc='sampling'):
        cond_sp = cond_spatial_all[i:i+1].to(device)
        cond_nu = nu_all[i:i+1].unsqueeze(1).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            pred_lr = diffusion.ddim_sample(cond_sp, cond_nu, num_steps=args.ddim_steps)
        pred_hr = F.interpolate(pred_lr, size=args.orig_L, mode='linear', align_corners=False)
        preds[i, 1:, :] = pred_hr.squeeze(0).cpu().float()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out = {
        'preds':   preds,
        'gts':     trajectories,
        'nu':      nu_all,
        'x_coord': data.get('x_coord'),
        't_coord': data.get('t_coord'),
        'model':   'AveragePooling_DiT (k=4)',
        'ckpt':    args.ckpt_path,
        'ddim_steps': args.ddim_steps,
        'seed':    args.seed,
    }
    torch.save(out, args.output_path)
    print(f"Saved: {args.output_path}")
    print(f"preds shape: {preds.shape}")


if __name__ == '__main__':
    main()
