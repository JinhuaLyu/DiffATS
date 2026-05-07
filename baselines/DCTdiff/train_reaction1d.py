
import os
import builtins
import time
from datetime import timedelta

import accelerate
import ml_collections
import numpy as np
import torch
from absl import app, flags, logging
from accelerate import InitProcessGroupKwargs
from ml_collections import config_flags
from torch import multiprocessing as mp
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import sde
import utils
from datasets_reaction import Reaction1D
from libs.uvit_pde import UViTReaction
from sde_pde import CondScoreModel, euler_maruyama_cond
from DCT_utils_1d import (
    DCT2DBlocks,
    reverse_zigzag_order_2d,
    tokens_to_field,
)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def _build_nnet(cfg):
    return UViTReaction(**cfg.nnet)


def _init_train_state(config, device):
    nnet = _build_nnet(config)
    nnet_ema = _build_nnet(config)
    nnet_ema.eval()
    logging.info(f'nnet has {sum(p.numel() for p in nnet.parameters())} parameters')

    optimizer = utils.get_optimizer(list(nnet.parameters()), **config.optimizer)
    lr_scheduler = utils.get_lr_scheduler(optimizer, **config.lr_scheduler)
    state = utils.TrainState(
        optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
        nnet=nnet, nnet_ema=nnet_ema,
    )
    state.ema_update(0)
    state.to(device)
    return state


def _decode_tokens_to_field(tokens, ds):
    """tokens: (B, n_tokens, F). Returns field (B, T_pad, X) in physical units
    (in roughly [0, 1] for Fisher-KPP, after re-adding shift)."""
    rev = reverse_zigzag_order_2d(ds.B_t, ds.B_x)
    dct_op = DCT2DBlocks(ds.B_t, ds.B_x).to(tokens.device)
    toks = tokens * ds.Y_bound
    field = tokens_to_field(
        toks, ds.B_t, ds.B_x, ds.M_x, ds.low_freqs, rev,
        n_t=ds.n_t, n_x=ds.n_x, dct_op=dct_op,
    )
    return field * ds.u_scale + ds.shift


def _save_sample_plot(field, path, ds, n_show=4):
    """Save a quick visualization: rows = samples, generated u(t=1..200)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    field = field.detach().cpu().numpy()                  # (B, T_pad, X)
    n_show = min(n_show, field.shape[0])
    fig, axes = plt.subplots(n_show, 1, figsize=(8, 2.0 * n_show), squeeze=False)
    for i in range(n_show):
        gen = field[i, ds.gen_start : ds.gen_start + 200]  # (200, X)
        ax = axes[i, 0]
        im = ax.imshow(gen, aspect='auto', origin='lower', cmap='viridis',
                       vmin=0.0, vmax=1.0)
        ax.set_ylabel(f'sample {i}')
        ax.set_xticks([])
        if i == 0:
            ax.set_title('generated u(t,x), rows = t=1..200, cols = x=0..1')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def LSimple_masked_reaction(score_model, x0, ic_clean, ic_mask,
                            cond_kwargs, pred='noise_pred', reweight=None):
    """Diffusion training loss with two scalar conditions, restricted to
    non-IC tokens. Mirrors sde_pde.LSimple_masked but threads arbitrary
    conditioning kwargs (e.g. nu and rho) through to the score network."""
    t, noise, xt = score_model.sde.sample(x0)

    if score_model.n_ic_tokens > 0:
        xt = xt.clone()
        xt[:, : score_model.n_ic_tokens] = ic_clean

    if pred == 'noise_pred':
        pred_t = score_model.noise_pred(xt, t, ic_clean=ic_clean, **cond_kwargs)
        diff = (noise - pred_t).pow(2)
    elif pred == 'x0_pred':
        x0_pred = score_model.x0_pred(xt, t, ic_clean=ic_clean, **cond_kwargs)
        diff = (x0 - x0_pred).pow(2)
    else:
        raise NotImplementedError(pred)

    if reweight is not None:
        diff = diff * reweight                                         # (B, L, F)

    keep = (~ic_mask).to(diff.device).view(1, -1, 1).float()           # (1, L, 1)
    n_keep = keep.sum() * diff.shape[2]
    return (diff * keep).flatten(1).sum(dim=-1) / n_keep                # (B,)


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True

    mp.set_start_method('spawn', force=True)
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    accelerator = accelerate.Accelerator(
        mixed_precision=config.train.get('mixed_precision', 'no'),
        kwargs_handlers=[process_group_kwargs],
    )
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'rank {accelerator.process_index} on {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes
    logging.info(f'{accelerator.num_processes} GPUs x batch {mini_batch_size}')

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args, **kwargs: None

    dataset = Reaction1D(**config.dataset)
    train_ds = dataset.get_split('train')
    train_loader = DataLoader(
        train_ds, batch_size=mini_batch_size, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=False, persistent_workers=True,
    )
    logging.info(f'train trajectories: {len(train_ds)}')
    logging.info(
        f'train trajectories={len(train_ds)} '
        f'global_batch={config.train.batch_size} '
        f'total_steps={config.train.n_steps}'
    )

    train_state = _init_train_state(config, device)
    nnet, nnet_ema, optimizer, train_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_loader,
    )
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    if config.train.get('use_reweight', True):
        per_freq_std = np.array(config.dataset.per_freq_std)
        rw = per_freq_std / (per_freq_std.sum() / per_freq_std.shape[0])
        reweight = torch.from_numpy(rw).float().to(device)         # (F,)
        assert reweight.shape[0] == train_ds.feature_dim
        logging.info(f'using per-frequency reweight (mean=1, max={reweight.max():.2f})')
    else:
        reweight = None
        logging.info('reweight disabled: uniform per-component loss')

    n_ic = config.dataset.n_ic_tokens

    score_model = CondScoreModel(
        nnet, pred=config.pred,
        sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale),
        n_ic_tokens=n_ic,
    )
    score_model_ema = CondScoreModel(
        nnet_ema, pred=config.pred,
        sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale),
        n_ic_tokens=n_ic,
    )

    def get_data_generator():
        while True:
            for batch in tqdm(train_loader, disable=not accelerator.is_main_process,
                              desc='epoch'):
                yield batch

    data_gen = get_data_generator()

    def train_step(batch):
        x0, nu, rho, ic_mask = batch                               # ic_mask: (B, L)
        ic_clean = x0[:, :n_ic]                                    # (B, n_ic, F)
        ic_mask_1d = ic_mask[0]                                    # (L,)
        optimizer.zero_grad()
        loss = LSimple_masked_reaction(
            score_model, x0, ic_clean=ic_clean, ic_mask=ic_mask_1d,
            cond_kwargs={'nu': nu, 'rho': rho},
            pred=config.pred, reweight=reweight,
        )
        m = {'loss': accelerator.gather(loss.detach()).mean()}
        accelerator.backward(loss.mean())
        if 'grad_clip' in config and config.grad_clip > 0:
            accelerator.clip_grad_norm_(nnet.parameters(), max_norm=config.grad_clip)
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=optimizer.param_groups[0]['lr'], **m)

    @torch.no_grad()
    def sample_visual(n=4):
        idxs = np.random.choice(len(train_ds), size=n, replace=False)
        x_target_list, nu_list, rho_list, ic_mask_list = [], [], [], []
        for i in idxs:
            xt, nu_v, rho_v, m = train_ds[int(i)]
            x_target_list.append(xt)
            nu_list.append(nu_v)
            rho_list.append(rho_v)
            ic_mask_list.append(m)
        x_target = torch.stack(x_target_list).to(device)           # (n, L, F)
        nu = torch.stack(nu_list).to(device)
        rho = torch.stack(rho_list).to(device)
        ic_clean = x_target[:, :n_ic]

        x_init = torch.randn_like(x_target)
        rsde = sde.ODE(score_model_ema)
        x_gen = euler_maruyama_cond(
            rsde, x_init, sample_steps=config.sample.sample_steps,
            n_ic_tokens=n_ic, ic_clean=ic_clean, nu=nu, rho=rho,
        )
        field_gen = _decode_tokens_to_field(x_gen, train_ds)
        return field_gen

    t_start = time.time()
    last_log_t = t_start
    last_log_step = train_state.step

    logging.info(f'start fitting at step={train_state.step}, mp={config.mixed_precision}')
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_gen))
        metrics = train_step(batch)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            now = time.time()
            dt = max(now - last_log_t, 1e-6)
            ds_steps = train_state.step - last_log_step
            steps_per_sec = ds_steps / dt
            steps_left = config.train.n_steps - train_state.step
            eta_h = steps_left / max(steps_per_sec, 1e-6) / 3600.0
            metrics['s/step'] = 1.0 / max(steps_per_sec, 1e-6)
            metrics['eta_h'] = eta_h
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            last_log_t = now
            last_log_step = train_state.step

        if (accelerator.is_main_process
                and train_state.step % config.train.eval_interval == 0):
            try:
                field = sample_visual(n=4)
                path = os.path.join(config.sample_dir, f'{train_state.step}.png')
                _save_sample_plot(field, path, train_ds)
                logging.info(f'saved viz -> {path}')
            except Exception as e:
                logging.warning(f'viz failed: {e}')
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if (train_state.step >= config.train.save_interval
                and train_state.step % config.train.save_interval == 0):
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()

    logging.info(f'done at step={train_state.step}')


def main(argv):
    config = FLAGS.config
    config.workdir = FLAGS.workdir or 'exp_reaction1d'
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')

    n_epochs = config.train.get('n_epochs', None)
    if n_epochs is not None:
        d = torch.load(config.dataset.path, map_location='cpu',
                       weights_only=False, mmap=True)
        n_train = d['tensor'].shape[0]
        del d
        steps_per_epoch = n_train // config.train.batch_size
        config.train.n_steps = n_epochs * steps_per_epoch
        print(f'[main] n_train={n_train} batch={config.train.batch_size} '
              f'steps/epoch={steps_per_epoch} epochs={n_epochs} '
              f'-> total_steps={config.train.n_steps}')

    train(config)


if __name__ == '__main__':
    app.run(main)
