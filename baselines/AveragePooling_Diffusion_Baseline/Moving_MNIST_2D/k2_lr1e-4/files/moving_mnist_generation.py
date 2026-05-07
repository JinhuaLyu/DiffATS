import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moving_mnist_dit_model import MNISTDiT, GaussianDiffusion


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        type=str, required=True,
                   help='Path to latest.pt checkpoint')
    p.add_argument('--out',         type=str, required=True,
                   help='Output .pt path for generated videos (N,T,H,W) uint8')
    p.add_argument('--n_generate',  type=int, default=10000,
                   help='Number of videos to generate')
    p.add_argument('--batch_size',  type=int, default=64,
                   help='Generation batch size (reduce if OOM)')
    p.add_argument('--ddim_steps',  type=int, default=250,
                   help='DDIM sampling steps')
    p.add_argument('--eta',         type=float, default=0.0,
                   help='DDIM eta (0 = deterministic)')
    # model hparams (must match training)
    p.add_argument('--hidden_dim',  type=int, default=512)
    p.add_argument('--num_layers',  type=int, default=12)
    p.add_argument('--num_heads',   type=int, default=8)
    p.add_argument('--spatial_size',type=int, default=32,
                   help='Spatial size used during training (64 // pool_k)')
    p.add_argument('--num_frames',  type=int, default=20)
    p.add_argument('--diff_timesteps', type=int, default=1000)
    return p.parse_args()


def load_model(args, device):
    print(f"[1/4] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)

    # support both ckpt['model'] and bare state_dict
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    # override hparams from checkpoint if available
    saved_args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    hidden_dim  = saved_args.get('hidden_dim',  args.hidden_dim)
    num_layers  = saved_args.get('num_layers',  args.num_layers)
    num_heads   = saved_args.get('num_heads',   args.num_heads)
    pool_k      = saved_args.get('pool_k',      2)
    spatial     = 64 // pool_k   # 32
    num_frames  = saved_args.get('num_frames',  args.num_frames)
    diff_ts     = saved_args.get('diff_timesteps', args.diff_timesteps)

    print(f"    spatial={spatial}x{spatial}, hidden={hidden_dim}, "
          f"layers={num_layers}, heads={num_heads}, T={diff_ts}")

    model = MNISTDiT(
        spatial_size=spatial,
        num_frames=num_frames,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    diffusion = GaussianDiffusion(model, timesteps=diff_ts)
    diffusion.load_state_dict(
        {k.replace('model.', 'model.'): v for k, v in state_dict.items()},
        strict=False
    )
    # load model weights only 
    model.load_state_dict(state_dict)
    diffusion = GaussianDiffusion(model, timesteps=diff_ts).to(device)
    diffusion.eval()

    epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    step  = ckpt.get('global_step', '?') if isinstance(ckpt, dict) else '?'
    print(f"    Loaded  epoch={epoch}  global_step={step}")
    return diffusion


@torch.no_grad()
def generate(diffusion, args, device):
    n_total  = args.n_generate
    bs       = args.batch_size
    n_batches = (n_total + bs - 1) // bs

    all_videos = []
    t0 = time.time()

    print(f"[2/4] DDIM {args.ddim_steps}-step generation  "
          f"({n_total} videos, batch={bs})...")

    for i in tqdm(range(n_batches), desc="  generating", ncols=90):
        actual_bs = min(bs, n_total - i * bs)

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            videos_lr = diffusion.ddim_sample(
                batch_size=actual_bs,
                num_steps=args.ddim_steps,
                eta=args.eta,
            )  # [B, 20, 32, 32]  float, [-1, 1]

        # bilinear upsample 32x32 to 64x64 
        B, T, H, W = videos_lr.shape
        videos_hr = F.interpolate(
            videos_lr.reshape(B * T, 1, H, W),   # treat each frame indep.
            size=(64, 64),
            mode='bilinear',
            align_corners=False,
        ).reshape(B, T, 64, 64)                   # [B, 20, 64, 64]

        #  denorm [-1,1] to uint8 [0,255]
        videos_uint8 = ((videos_hr.clamp(-1, 1) + 1) / 2 * 255) \
                       .round().to(torch.uint8).cpu()

        all_videos.append(videos_uint8)

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s  "
          f"({elapsed / n_total * 1000:.1f} ms/video)")

    return torch.cat(all_videos, dim=0)   # (N, T, 64, 64)  uint8


def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # load 
    diffusion = load_model(args, device)

    # generate
    videos = generate(diffusion, args, device)   # (10000, 20, 64, 64) uint8

    print(f"[3/4] Generated tensor: {tuple(videos.shape)}  dtype={videos.dtype}  "
          f"min={videos.min()}  max={videos.max()}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[4/4] Saving → {args.out}")
    torch.save(videos, args.out)
    print(f"    Saved  ({os.path.getsize(args.out)/1e9:.2f} GB)")
    print("Done.")


if __name__ == '__main__':
    main()
