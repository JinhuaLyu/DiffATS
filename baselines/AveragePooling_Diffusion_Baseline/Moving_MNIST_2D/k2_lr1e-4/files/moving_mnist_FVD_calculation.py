
import argparse
import json
import os
import time
import numpy as np
import torch
from scipy.linalg import sqrtm


# tensor loading 

def load_video_tensor(path, n=None, select='first', seed=0):
    print(f"Loading: {path}")
    obj = torch.load(path, map_location='cpu', weights_only=False)
    x   = obj if not isinstance(obj, dict) else obj['videos']

    assert x.dtype == torch.uint8, f"Expected uint8, got {x.dtype}"
    assert x.dim() == 4,           f"Expected 4D tensor, got {x.shape}"

    # auto-detect layout: if dim0 << dim1, it's (T, N, H, W)
    if x.shape[0] < x.shape[1] and x.shape[0] <= 64:
        x = x.permute(1, 0, 2, 3).contiguous()   # → (N, T, H, W)
        layout = "(T, N, H, W) → transposed"
    else:
        layout = "(N, T, H, W)"

    print(f"  layout={layout}  shape={tuple(x.shape)}  "
          f"dtype={x.dtype}  min={x.min()}  max={x.max()}")

    # subsample
    if n is not None and x.shape[0] > n:
        if select == 'random':
            g   = torch.Generator().manual_seed(seed)
            idx = torch.randperm(x.shape[0], generator=g)[:n]
            x   = x[idx]
            print(f"  random {n} samples (seed={seed})")
        else:
            x = x[:n]
            print(f"  first {n} samples")

    return x   # (N, T, H, W) uint8


# I3D feature extraction 

@torch.no_grad()
def extract_features(videos_NTHW: torch.Tensor,
                     model,
                     device: torch.device,
                     
    N, T, H, W = videos_NTHW.shape

    # probe output dimension
    probe = videos_NTHW[:1].to(device).float()
    probe = probe.unsqueeze(1).expand(1, 3, T, H, W)   # (1, 3, T, H, W)
    probe_out = model(probe, True, True, True)
    D = probe_out.shape[1]
    print(f"  I3D feature dim = {D}")

    feats = torch.empty((N, D), dtype=torch.float32)
    t0    = time.time()

    for i in range(0, N, batch_size):
        batch = videos_NTHW[i : i + batch_size]
        b     = batch.shape[0]
        x     = batch.to(device).float()
        x     = x.unsqueeze(1).expand(b, 3, T, H, W)   # (B, 3, T, H, W)
        # TorchScript I3D signature: forward(x, rescale, resize, return_features)
        f     = model(x, True, True, True)
        feats[i : i + b] = f.float().cpu()

        if (i // batch_size) % 20 == 0:
            print(f"  feats {i+b:6d}/{N}  elapsed={time.time()-t0:.1f}s",
                  flush=True)

    print(f"  done  total={time.time()-t0:.1f}s")
    return feats.numpy().astype(np.float64)


# Fréchet distance 

def compute_fvd(feats_r: np.ndarray, feats_g: np.ndarray):
    mu_r, mu_g       = feats_r.mean(0), feats_g.mean(0)
    sigma_r, sigma_g = np.cov(feats_r, rowvar=False), np.cov(feats_g, rowvar=False)

    diff    = mu_r - mu_g
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)

    if np.iscomplexobj(covmean):
        imag_max = np.max(np.abs(covmean.imag))
        if imag_max > 1e-3:
            print(f"  WARNING: sqrtm imag max = {imag_max:.3e}")
        covmean = covmean.real

    fvd    = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    return fvd, float(np.linalg.norm(diff) ** 2), float(np.trace(sigma_r)), float(np.trace(sigma_g))


# main 

def get_args():
    p = argparse.ArgumentParser(description="FVD for Moving MNIST generation pipeline")
    p.add_argument('--real',        type=str, required=True,
                   default='/anvil/scratch/x-ezhou1/physics_datasets/data/moving_mnist/moving_mnist_20k_2slow.pt')
    p.add_argument('--gen',         type=str, required=True,
                   help='Path to generated_10k.pt from MovingMnist_generation.py')
    p.add_argument('--i3d',         type=str,
                   default='/home/x-jlyu5/.cache/fvd/i3d_torchscript.pt')
    p.add_argument('--out',         type=str, required=True,
                   help='Output JSON report path')
    p.add_argument('--n',           type=int, default=10000,
                   help='Number of videos to use from each set')
    p.add_argument('--batch',       type=int, default=32,
                   help='I3D extraction batch size')
    p.add_argument('--real_select', choices=['first', 'random'], default='random',
                   help='How to subsample real videos')
    p.add_argument('--seed',        type=int, default=42)
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # load I3D
    print(f"\nLoading I3D: {args.i3d}")
    assert os.path.exists(args.i3d), \
        f"I3D weights not found at {args.i3d}\n" \
        f"Download from: https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
    model = torch.jit.load(args.i3d, map_location=device).eval()

    # load videos
    print("\n--- Real videos ---")
    real = load_video_tensor(args.real, n=args.n,
                             select=args.real_select, seed=args.seed)
    print("\n--- Generated videos ---")
    gen  = load_video_tensor(args.gen,  n=args.n,
                             select='first', seed=args.seed)
    assert real.shape == gen.shape, \
        f"Shape mismatch: real={tuple(real.shape)}  gen={tuple(gen.shape)}"
    assert real.shape[0] == args.n, \
        f"Expected {args.n} videos, got {real.shape[0]}"

    print(f"\nBoth tensors: {tuple(real.shape)}  ✓")

    # extract features
    print("\n[1/3] Extracting real features...")
    feats_r = extract_features(real, model, device, args.batch)

    print("\n[2/3] Extracting generated features...")
    feats_g = extract_features(gen,  model, device, args.batch)

    # compute FVD
    print("\n[3/3] Computing FVD...")
    fvd, mean_sq, tr_r, tr_g = compute_fvd(feats_r, feats_g)

    print(f"\n{'='*45}")
    print(f"  FVD                    = {fvd:.4f}")
    print(f"  ||mu_r - mu_g||^2      = {mean_sq:.4f}")
    print(f"  tr(Sigma_real)         = {tr_r:.4f}")
    print(f"  tr(Sigma_gen)          = {tr_g:.4f}")
    print(f"{'='*45}")

    # save 
    report = {
        'fvd':               fvd,
        'mean_sq_diff':      mean_sq,
        'trace_sigma_real':  tr_r,
        'trace_sigma_gen':   tr_g,
        'n_samples':         int(args.n),
        'frame_count':       int(real.shape[1]),
        'frame_size':        [int(real.shape[2]), int(real.shape[3])],
        'batch_size':        int(args.batch),
        'real_path':         os.path.abspath(args.real),
        'gen_path':          os.path.abspath(args.gen),
        'real_select':       args.real_select,
        'real_seed':         int(args.seed) if args.real_select == 'random' else None,
        'i3d_ckpt':          os.path.abspath(args.i3d),
        'feature_dim':       int(feats_r.shape[1]),
        'device':            str(device),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {args.out}")


if __name__ == '__main__':
    main()
