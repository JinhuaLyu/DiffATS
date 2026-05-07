import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
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

    def count_parameters(self):
        return sum(p.numel() * (2 if p.is_complex() else 1)
                   for p in self.parameters() if p.requires_grad)

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


class MovingMNISTDataset(Dataset):
    def __init__(self, data, seq_len=20):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :self.seq_len].to(torch.float32).div_(255.0)


def _load_tensor(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.shape[0] < raw.shape[1]:
        raw = raw.permute(1, 0, 2, 3).contiguous()
    return raw


def make_loaders(data_path, batch_size, num_workers, seq_len, val_samples=2000):
    data = _load_tensor(data_path)
    full = MovingMNISTDataset(data, seq_len=seq_len)
    train_loader = DataLoader(full, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    val_ds = Subset(full, range(min(val_samples, len(full))))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)
    return train_loader, val_loader


T_IN = 1
T_OUT = 19
SEQ_LEN = 20

MODEL_CFG = dict(t_in=T_IN, n_modes=(16, 16), hidden_channels=132, n_layers=8)

BATCH_SIZE = 16
LR = 1e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SS_MAX = 0.8
SS_RAMP = 0.5
EPOCHS = 200
VAL_EVERY = 5

DATA_ROOT = "/gpfs/home/bkx8728/Tensor_factor/moving_mnist/moving_mnist_20k_2slow.pt"
CKPT_DIR = "/projects/e30514/bkx8728/checkpoints"
NUM_WORKERS = 4
VAL_SAMPLES = 2_000


def train_epoch(model, train_loader, optimizer, scheduler, ss_prob, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device, non_blocking=True)
        B = batch.shape[0]
        optimizer.zero_grad(set_to_none=True)
        context = batch[:, :T_IN]
        loss = torch.tensor(0.0, device=device)
        for t in range(T_OUT):
            target = batch[:, T_IN + t].unsqueeze(1)
            pred = model(context)
            loss = loss + F.mse_loss(pred, target)
            if ss_prob > 0.0:
                mask = torch.rand(B, 1, 1, 1, device=device) < ss_prob
                next_frame = torch.where(mask, pred.detach(), target)
            else:
                next_frame = target
            context = torch.cat([context[:, 1:], next_frame], dim=1)
        loss = loss / T_OUT
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, device, n_batches=20):
    model.eval()
    step_losses = torch.zeros(T_OUT)
    total = 0
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        batch = batch.to(device, non_blocking=True)
        B = batch.shape[0]
        context = batch[:, :T_IN]
        for t in range(T_OUT):
            target = batch[:, T_IN + t].unsqueeze(1)
            pred = model(context)
            step_losses[t] += F.mse_loss(pred, target).item() * B
            context = torch.cat([context[:, 1:], pred], dim=1)
        total += B
    step_losses /= total
    model.train()
    return step_losses.mean().item()


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA_ROOT)
    parser.add_argument("--ckpt-dir", type=str, default=CKPT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_loader, val_loader = make_loaders(
        args.data, batch_size=args.batch_size,
        num_workers=args.workers, seq_len=SEQ_LEN, val_samples=VAL_SAMPLES,
    )

    model = FNO2DAutoregressive(**MODEL_CFG).to(device)
    n_params = model.count_parameters()
    print(f"params={n_params:,}  ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=LR_MIN)

    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        print(f"resumed from epoch {ckpt['epoch']}  best_val={best_val:.6f}")

    for epoch in range(start_epoch, args.epochs + 1):
        ss_prob = min(epoch / (args.epochs * SS_RAMP), 1.0) * SS_MAX
        train_mse = train_epoch(model, train_loader, optimizer, scheduler, ss_prob, device)
        print(f"epoch={epoch}  train_mse={train_mse:.5f}  ss={ss_prob:.2f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % VAL_EVERY == 0 or epoch == args.epochs:
            val_mse = validate(model, val_loader, device)
            print(f"  [val] val_mse={val_mse:.5f}")
            if val_mse < best_val:
                best_val = val_mse
                save_checkpoint(f"{args.ckpt_dir}/best.pt", model, optimizer, scheduler, epoch, best_val)
                print(f"  [best] saved  val_mse={best_val:.5f}")

    save_checkpoint(f"{args.ckpt_dir}/last.pt", model, optimizer, scheduler, args.epochs, best_val)


if __name__ == "__main__":
    main()
