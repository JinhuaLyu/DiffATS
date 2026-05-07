import argparse
import os
import time

import torch
import torch.nn as nn
from neuralop.models import FNO


class FNO2DAutoregressive(nn.Module):
    def __init__(self, t_in=1, n_modes=(16, 16), hidden_channels=132, n_layers=8):
        super().__init__()
        self.t_in = t_in
        self.fno = FNO(
            n_modes=n_modes, in_channels=t_in, out_channels=1,
            hidden_channels=hidden_channels, n_layers=n_layers,
            use_channel_mlp=True, channel_mlp_expansion=0.5,
            norm="instance_norm", non_linearity=nn.functional.gelu,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fno(x))

    @torch.no_grad()
    def autoregressive_rollout(self, context, steps):
        self.eval()
        buf = context.clone()
        preds = []
        for _ in range(steps):
            p = self(buf)
            preds.append(p)
            buf = torch.cat([buf[:, 1:], p], dim=1)
        return torch.cat(preds, dim=1)


def _load_tensor(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.shape[0] < raw.shape[1]:
        raw = raw.permute(1, 0, 2, 3).contiguous()
    return raw

T_IN = 1
T_OUT = 19
SEQ_LEN = 20
MODEL_CFG = dict(t_in=T_IN, n_modes=(16, 16), hidden_channels=132, n_layers=8)


def load_model(ckpt_path: str, device: torch.device) -> FNO2DAutoregressive:
    model = FNO2DAutoregressive(**MODEL_CFG).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"]
    if isinstance(state, dict):
        state.pop("_metadata", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"load_state_dict: missing={missing}  unexpected={unexpected}")
    model.eval()
    print(f"loaded {ckpt_path}  step={ckpt.get('step')}  best_val={ckpt.get('best_val')}")
    return model


@torch.no_grad()
def rollout_batch(model, batch_seq: torch.Tensor) -> torch.Tensor:
    """batch_seq: (B, SEQ_LEN, H, W) float in [0,1]. Returns (B, 20, H, W): t=0 real + t=1..19 predicted."""
    context = batch_seq[:, :T_IN]                        # (B, 1, H, W) — frame t=0
    pred = model.autoregressive_rollout(context, T_OUT)  # (B, 19, H, W)
    full = torch.cat([context, pred], dim=1)              # (B, 20, H, W)
    return full.clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="/projects/e30514/bkx8728/checkpoints/best.pt")
    parser.add_argument("--data", type=str,
                        default="/gpfs/home/bkx8728/Tensor_factor/moving_mnist/moving_mnist_20k_2slow.pt")
    parser.add_argument("--out", type=str, required=True,
                        help="Path to save generated video tensor (.pt, uint8, (N,T,H,W)).")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    data = _load_tensor(args.data)  # (N, T, H, W) uint8
    n_total = data.shape[0]
    if args.n > n_total:
        raise ValueError(f"requested n={args.n} but data has only {n_total}")
    print(f"data: {tuple(data.shape)}  N_total={n_total}  using first n={args.n}")

    model = load_model(args.ckpt, device)

    out = torch.empty((args.n, SEQ_LEN, data.shape[-2], data.shape[-1]), dtype=torch.uint8)
    t0 = time.time()

    for i in range(0, args.n, args.batch):
        end = min(i + args.batch, args.n)
        batch = data[i:end, :SEQ_LEN].to(device, dtype=torch.float32).div_(255.0)
        full = rollout_batch(model, batch)  # (B, 20, H, W) in [0,1]
        full_u8 = (full.mul(255.0).round_().clamp_(0, 255)).to(torch.uint8).cpu()
        out[i:end] = full_u8

        if (i // args.batch) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  rollout {end:5d}/{args.n}  elapsed={elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"rollout done  total={elapsed:.1f}s  shape={tuple(out.shape)}  dtype={out.dtype}")

    torch.save(out, args.out)
    print(f"wrote {args.out}  size={os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
