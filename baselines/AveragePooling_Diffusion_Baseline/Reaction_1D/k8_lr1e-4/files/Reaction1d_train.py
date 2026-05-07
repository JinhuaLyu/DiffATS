import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import time
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import wandb
from Reaction1d_avgpooling_dataset import Reaction1dDataset
from Reaction1d_DiT_Model import TrajDiT1D, GaussianDiffusion


# Logger

def setup_logger(log_dir, job_id=None):
    os.makedirs(log_dir, exist_ok=True)
    ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f"_{job_id}" if job_id else ""
    path   = os.path.join(log_dir, f'train_{ts}{suffix}.log')

    logger = logging.getLogger('reaction1d_train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt      = logging.Formatter('[\033[34m%(asctime)s\033[0m] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fmt_file = logging.Formatter('[%(asctime)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    sh = logging.StreamHandler();   sh.setFormatter(fmt)
    fh = logging.FileHandler(path); fh.setFormatter(fmt_file)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info(f"Log file: {path}")
    return logger


# Sample visualisation, wandb

@torch.no_grad()
def make_sample_plot(diffusion, dataset, device,
                     num_samples=3, num_ddim_steps=50, orig_L=1024):
    diffusion.eval()
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 2, figsize=(14, num_samples * 3))
    if num_samples == 1:
        axes = axes[None, :]

    show_frames = [0, 49, 99, 149, 199]

    for row, idx in enumerate(indices):
        sample       = dataset[idx]
        cond_spatial = sample['cond_spatial'].unsqueeze(0).to(device)
        cond_params  = sample['cond_params'].unsqueeze(0).to(device)
        target       = sample['target'].unsqueeze(0).to(device)

        with torch.amp.autocast('cuda'):
            pred_lr = diffusion.ddim_sample(
                cond_spatial, cond_params, num_steps=num_ddim_steps
            )

        pred_hr = torch.nn.functional.interpolate(
            pred_lr, size=orig_L, mode='linear', align_corners=False
        )
        gt_hr = torch.nn.functional.interpolate(
            target, size=orig_L, mode='linear', align_corners=False
        )

        nu_val  = cond_params[0, 0].item()
        rho_val = cond_params[0, 1].item()

        ax = axes[row, 0]
        for f in show_frames:
            ax.plot(gt_hr[0, f].cpu().numpy(), alpha=0.7, label=f't={f+1}')
        ax.set_title(f'GT  nu={nu_val:.5f}  rho={rho_val:.2f}', fontsize=9)
        ax.legend(fontsize=7)

        ax = axes[row, 1]
        for f in show_frames:
            ax.plot(pred_hr[0, f].cpu().numpy(), alpha=0.7, label=f't={f+1}')
        ax.set_title(f'Pred  nu={nu_val:.5f}  rho={rho_val:.2f}', fontsize=9)
        ax.legend(fontsize=7)

    plt.tight_layout()
    diffusion.train()
    return fig


# Checkpoint helpers

def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"  Saved: {path}")


def load_checkpoint(path, model, optimizer, scaler):
    if not os.path.exists(path):
        print(f"[Resume] No checkpoint at {path}, starting fresh.")
        return 0, 0
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    gs = ckpt.get('global_step', 0)
    print(f"[Resume] epoch={ckpt['epoch']}  step={gs}")
    return ckpt['epoch'], gs


# One epoch (AMP)

def train_one_epoch(diffusion, loader, optimizer, scaler, device,
                    grad_clip, epoch, args, logger, global_step):
    diffusion.train()
    total_loss  = 0.0
    epoch_start = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                ncols=110, mininterval=1.0)

    for step, batch in enumerate(pbar):
        cond_spatial = batch['cond_spatial'].to(device, non_blocking=True)
        cond_params  = batch['cond_params'].to(device,  non_blocking=True)
        target       = batch['target'].to(device,       non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            loss = diffusion.training_loss(target, cond_spatial, cond_params)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(diffusion.model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_val    = loss.item()
        total_loss += loss_val
        global_step += 1
        pbar.set_postfix(loss=f"{loss_val:.4f}")

        if global_step % args.log_every_steps == 0:
            elapsed = time.time() - epoch_start
            sps     = (step + 1) / elapsed
            logger.info(f"(step={global_step:07d}) "
                        f"loss={loss_val:.4f}  sps={sps:.2f}")
            if not args.no_wandb:
                wandb.log({
                    'train/loss':          loss_val,
                    'train/steps_per_sec': sps,
                    'train/step':          global_step,
                }, step=global_step)

    avg_loss   = total_loss / len(loader)
    epoch_time = time.time() - epoch_start
    return avg_loss, global_step, epoch_time


# Args

def get_args():
    p = argparse.ArgumentParser(
        description='1D Reaction-Diffusion Conditional Diffusion Training'
    )

    p.add_argument('--data_path', type=str,
        default='/anvil/scratch/x-<user>/physics_datasets/data/reaction_1d/reaction_1d_train.pt')
    p.add_argument('--pool_k', type=int, default=8)

    # model
    p.add_argument('--hidden_dim',     type=int, default=512)
    p.add_argument('--num_layers',     type=int, default=12)
    p.add_argument('--num_heads',      type=int, default=8)
    p.add_argument('--diff_timesteps', type=int, default=1000)

    # training
    p.add_argument('--epochs',      type=int,   default=500)
    p.add_argument('--batch_size',  type=int,   default=64)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--num_workers', type=int,   default=8)
    p.add_argument('--grad_clip',   type=float, default=1.0)

    # logging & saving
    p.add_argument('--output_dir', type=str,
        default='/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/Reaction_1D/k8_fixlr')
    p.add_argument('--save_every',          type=int,  default=50)
    p.add_argument('--log_every_steps',     type=int,  default=100)
    p.add_argument('--sample_every_epochs', type=int,  default=10)
    p.add_argument('--num_sample_plots',    type=int,  default=3)
    p.add_argument('--wandb_project',  type=str, default='reaction1d_k8')
    p.add_argument('--wandb_name',     type=str, default=None)
    p.add_argument('--no_wandb',       action='store_true')
    p.add_argument('--resume', type=str, default=None)

    return p.parse_args()


# Main

def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    log_dir  = os.path.join(args.output_dir, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    job_id = os.environ.get('SLURM_JOB_ID', None)
    logger = setup_logger(log_dir, job_id)
    logger.info(f"Device : {device}")
    logger.info(f"Args   : {vars(args)}")
    logger.info(f"LR     : fixed {args.lr} (no scheduler)")

    # model
    L_lr = 1024 // args.pool_k   # 128 for pool_k=8
    logger.info(f"Building TrajDiT1D (L_lr={L_lr}, tokens={L_lr+1})...")

    model = TrajDiT1D(
        L_lr=L_lr,
        num_frames=200,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)

    diffusion = GaussianDiffusion(model, timesteps=args.diff_timesteps).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters : {n_params/1e6:.2f}M")

    # optimizer & scaler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler('cuda')

    # resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, scaler
        )

    # wandb
    run_name     = args.wandb_name or \
        f"reaction1d_k{args.pool_k}_dim{args.hidden_dim}_fixlr{args.lr}"
    wandb_entity = os.environ.get('WANDB_ENTITY', None)
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=vars(args),
            resume='allow',
        )

    # dataset
    logger.info("Loading dataset...")
    train_ds = Reaction1dDataset(
        data_path=args.data_path,
        pool_k=args.pool_k,
    )
    logger.info(f"Train: {len(train_ds):,} trajectories")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # training loop
    logger.info(f"Training epoch {start_epoch+1} -> {args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        train_loss, global_step, epoch_time = train_one_epoch(
            diffusion, train_loader, optimizer, scaler, device,
            args.grad_clip, epoch, args, logger, global_step
        )

        lr = args.lr
        logger.info(
            f"Epoch [{epoch+1:4d}/{args.epochs}]  "
            f"train={train_loss:.4f}  lr={lr:.2e}  time={epoch_time:.1f}s"
        )

        if not args.no_wandb:
            wandb.log({
                'train/loss_epoch': train_loss,
                'train/time_epoch': epoch_time,
                'lr':               lr,
                'epoch':            epoch + 1,
            }, step=global_step)

        if not args.no_wandb and (epoch + 1) % args.sample_every_epochs == 0:
            logger.info("Generating sample plot...")
            try:
                fig = make_sample_plot(
                    diffusion, train_ds, device,
                    num_samples=args.num_sample_plots,
                    num_ddim_steps=50,
                )
                wandb.log({'samples/preview': wandb.Image(fig),
                           'epoch': epoch + 1}, step=global_step)
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Sample plot failed: {e}")

        # checkpoint
        ckpt = {
            'epoch':       epoch + 1,
            'global_step': global_step,
            'model':       model.state_dict(),
            'optimizer':   optimizer.state_dict(),
            'scaler':      scaler.state_dict(),
            'args':        vars(args),
        }

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(ckpt,
                os.path.join(ckpt_dir, f'epoch_{epoch+1:04d}.pt'))

        save_checkpoint(ckpt, os.path.join(ckpt_dir, 'latest.pt'))

    save_checkpoint(ckpt, os.path.join(ckpt_dir, 'final.pt'))
    logger.info("Done!")
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
