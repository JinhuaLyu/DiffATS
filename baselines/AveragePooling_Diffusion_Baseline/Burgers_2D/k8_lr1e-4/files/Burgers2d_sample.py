import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Burgers2d_DiT_Model import TrajDiT, GaussianDiffusion


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_path', type=str,
        default='${DATA_ROOT}/pde_samples/2d_burgers/pde_samples_10.pt')
    p.add_argument('--ckpt_path', type=str,
        default='${DATA_ROOT}/baseline_checkpoint/average_pooling/Burgers2d_k8_latest.pt')
    p.add_argument('--output_path', type=str,
        default='${DATA_ROOT}/pde_samples_generated/2dburgers/AveragePooling/avgpool_burgers2d_generated.pt')
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
    # NOTE per data['note']: only ux included; uy dropped. Use as single channel.
    trajectories = data['tensor']    # [10, 201, 128, 128]
    N = trajectories.shape[0]
    print(f"Num test trajectories: {N}")
    print(f"data note: {data.get('note')}")

    preds = torch.zeros(N, 201, args.orig_size, args.orig_size, dtype=torch.float32)
    preds[:, 0] = trajectories[:, 0]

    for i in tqdm(range(N), desc='sampling'):
        field = trajectories[i].to(device)             # [201, 128, 128]
        cond_lr = F.avg_pool2d(
            field[0].unsqueeze(0).unsqueeze(0), args.pool_k
        )                                              # [1, 1, 16, 16]
        with torch.no_grad(), torch.amp.autocast('cuda'):
            traj_lr = diffusion.ddim_sample(cond_lr, num_steps=args.ddim_steps)
        # traj_lr: [1, 200, 16, 16]
        traj_hr = F.interpolate(
            traj_lr.squeeze(0).unsqueeze(1),
            size=(args.orig_size, args.orig_size),
            mode='bilinear', align_corners=False,
        ).squeeze(1)                                   # [200, 128, 128]
        preds[i, 1:] = traj_hr.cpu().float()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out = {
        'pred':  preds,
        'gt':    trajectories,
        'nu':    data.get('nu'),
        'convection_delta': data.get('convection_delta'),
        'diffusion_gamma':  data.get('diffusion_gamma'),
        'note':  'AveragePooling baseline; ux channel only (matches test data note)',
        'model': 'AveragePooling_DiT (k=8)',
        'ckpt':  args.ckpt_path,
        'ddim_steps': args.ddim_steps,
        'seed':  args.seed,
    }
    torch.save(out, args.output_path)
    print(f"Saved: {args.output_path}")
    print(f"pred shape: {preds.shape}")


if __name__ == '__main__':
    main()
