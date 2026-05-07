
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from CelebA_DiT_Model import CelebADiT, GaussianDiffusion


def get_args():
    p = argparse.ArgumentParser(description='CelebA-HQ Sampling')

    p.add_argument('--checkpoint',    type=str,
        default='/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/CelebA_HQ/k4_patch8_fixlr_Epoch878/checkpoints/latest.pt')
    p.add_argument('--output_dir',    type=str,
        default='/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/CelebA_HQ/k4_patch8_fixlr_Epoch878/samples')

    # model arch 
    p.add_argument('--pool_k',        type=int, default=4)
    p.add_argument('--orig_size',     type=int, default=1024)
    p.add_argument('--patch_size',    type=int, default=8)
    p.add_argument('--hidden_dim',    type=int, default=768)
    p.add_argument('--num_layers',    type=int, default=12)
    p.add_argument('--num_heads',     type=int, default=12)
    p.add_argument('--diff_timesteps',type=int, default=1000)

    # sampling
    p.add_argument('--num_samples',   type=int, default=10000)
    p.add_argument('--batch_size',    type=int, default=16)
    p.add_argument('--ddim_steps',    type=int, default=250)
    p.add_argument('--eta',           type=float, default=0.0)
    p.add_argument('--upsample',      action='store_true', default=True,
        help='Upsample generated 256x256 to orig_size (1024x1024)')

    return p.parse_args()


@torch.no_grad()
def sample_batch(diffusion, batch_size, ddim_steps, eta, orig_size, upsample, device):
    diffusion.eval()
    with torch.amp.autocast('cuda'):
        gen_lr = diffusion.ddim_sample(
            batch_size=batch_size,
            num_steps=ddim_steps,
            eta=eta,
        )  # [B, 3, 256, 256]

    if upsample:
        gen = F.interpolate(
            gen_lr, size=(orig_size, orig_size),
            mode='bilinear', align_corners=False
        )  # [B, 3, 1024, 1024]
    else:
        gen = gen_lr

    # [-1, 1] -> [0, 255]
    gen = ((gen + 1) / 2).clamp(0, 1)
    gen = (gen * 255).byte().cpu()
    return gen  # [B, 3, H, W] uint8


def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device       = {device}")
    print(f"[INFO] checkpoint   = {args.checkpoint}")
    print(f"[INFO] output_dir   = {args.output_dir}")
    print(f"[INFO] num_samples  = {args.num_samples}")
    print(f"[INFO] ddim_steps   = {args.ddim_steps}")
    print(f"[INFO] batch_size   = {args.batch_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    # build model
    lr_size = args.orig_size // args.pool_k  # 256
    print(f"[INFO] building CelebADiT (lr_size={lr_size}) ...")

    model = CelebADiT(
        lr_size=lr_size,
        patch_size=args.patch_size,
        in_channels=3,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)

    diffusion = GaussianDiffusion(model, timesteps=args.diff_timesteps).to(device)

    # load checkpoint 
    print(f"[INFO] loading checkpoint ...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"[INFO] loaded epoch={ckpt.get('epoch', '?')}  step={ckpt.get('global_step', '?')}")

    # sampling 
    n_generated = 0
    n_batches   = (args.num_samples + args.batch_size - 1) // args.batch_size

    print(f"[INFO] starting sampling ({n_batches} batches) ...")
    pbar = tqdm(total=args.num_samples, desc='Sampling', unit='img')

    for i in range(n_batches):
        this_batch = min(args.batch_size, args.num_samples - n_generated)
        imgs = sample_batch(
            diffusion, this_batch, args.ddim_steps,
            args.eta, args.orig_size, args.upsample, device
        )  # [B, 3, H, W] uint8

        for j in range(imgs.shape[0]):
            img_np  = imgs[j].permute(1, 2, 0).numpy()
            img_pil = Image.fromarray(img_np)
            fname   = os.path.join(args.output_dir, f'sample_{n_generated:05d}.png')
            img_pil.save(fname)
            n_generated += 1

        pbar.update(imgs.shape[0])

    pbar.close()
    print(f"[DONE] {n_generated} images saved to {args.output_dir}")


if __name__ == '__main__':
    main()
