import json
import os
import builtins
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import accelerate
import ml_collections
from absl import logging
from accelerate import InitProcessGroupKwargs
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils._pytree import tree_map
from tqdm.auto import tqdm

import sde
import utils
from datasets import get_dataset
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from DCT_utils import zigzag_order_3d, reverse_zigzag_order_3d


STATS_PATH = Path(__file__).parent / 'karman_stats_3d.json'


def _load_stats(config):
    if not STATS_PATH.exists():
        raise FileNotFoundError(
            f'{STATS_PATH} not found — run karman_3d_statis.py first'
        )
    stats = json.loads(STATS_PATH.read_text())
    config.dataset.Y_bound = [stats['Y_bound']]
    config.dataset.vor_std  = stats['vor_std']
    logging.info(
        f"Loaded stats from {STATS_PATH}: Y_bound={stats['Y_bound']:.6f}, "
        f"vor_std[:5]={stats['vor_std'][:5]}"
    )


def _peak_gpu_mb(device) -> float:
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    accelerator = accelerate.Accelerator(kwargs_handlers=[process_group_kwargs])
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    _load_stats(config)
    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes
    logging.info(f'use {accelerator.num_processes} GPUs with batch size {mini_batch_size}/GPU')

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    # wandb (optional)
    _wandb_run = None
    _wandb_cfg = config.get('wandb', None)
    if accelerator.is_main_process and _wandb_cfg is not None \
            and _wandb_cfg.get('mode', 'online') != 'disabled':
        try:
            import wandb as _wandb
            _wandb_run = _wandb.init(
                project=_wandb_cfg.get('project', 'DCTdiff-Karman'),
                name=_wandb_cfg.get('name', None) or os.path.basename(config.workdir),
                mode=_wandb_cfg.get('mode', 'online'),
                tags=list(_wandb_cfg.get('tags', [])) or None,
                dir=config.workdir,
                config=config.to_dict(),
                resume='allow',
            )
            logging.info(f'wandb run: {_wandb_run.url}')
        except Exception as e:
            logging.warning(f'wandb init failed ({e}); continuing without wandb')

    # Dataset
    dataset = get_dataset(**config.dataset)
    train_dataset = dataset.get_split(split='train', labeled=False)
    data_num_workers = int(config.dataset.get('num_workers', 4))
    train_dataset_loader = DataLoader(
        train_dataset,
        batch_size=mini_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=data_num_workers,
        pin_memory=False,
        persistent_workers=data_num_workers > 0,
    )
    logging.info(f'dataset samples: {len(train_dataset)}')

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader
    )
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    # Loss reweighting (3D zigzag std)
    bT, bH, bW = config.dataset.block_T, config.dataset.block_H, config.dataset.block_W
    reverse_order_3d = reverse_zigzag_order_3d(bT, bH, bW)

    std_values = np.array(config.dataset.vor_std)
    reweight = std_values[:config.dataset.low_freqs]
    reweight = reweight / (reweight.sum() / reweight.shape[0])
    reweight_by_std = torch.from_numpy(reweight).to(device=device).float()
    assert reweight_by_std.shape[0] == config.dataset.low_freqs

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    score_model     = sde.ScoreModel(nnet,     pred=config.pred, sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale))
    score_model_ema = sde.ScoreModel(nnet_ema, pred=config.pred, sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale))

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        if isinstance(_batch, (tuple, list)):
            x0, cond = _batch[0], _batch[1]
            loss = sde.LSimple(score_model, x0, pred=config.pred,
                               reweight=reweight_by_std, cond=cond)
        else:
            loss = sde.LSimple(score_model, _batch, pred=config.pred,
                               reweight=reweight_by_std)
        _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        accelerator.backward(loss.mean())
        if 'grad_clip' in config and config.grad_clip > 0:
            accelerator.clip_grad_norm_(nnet.parameters(), max_norm=config.grad_clip)
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        if device.type == 'cuda':
            _metrics['gpu_mem_mb'] = _peak_gpu_mb(device)
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def _dpm_sample(n_samples, cond=None):
        x_init = torch.randn(n_samples, *dataset.data_shape, device=device)
        noise_schedule = NoiseScheduleVP(schedule='linear', SNR_scale=config.dataset.SNR_scale)
        kwargs = dict(cond=cond) if cond is not None else {}
        model_fn = model_wrapper(score_model_ema.noise_pred, noise_schedule,
                                 time_input_type='0', model_kwargs=kwargs)
        dpm_solver = DPM_Solver(model_fn, noise_schedule)
        return dpm_solver.sample(x_init, steps=config.sample.sample_steps,
                                 eps=1e-4, adaptive_step_size=False, fast_version=True)

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            log_dict = dict(step=train_state.step, **metrics)
            logging.info(utils.dct2str(log_dict))
            if _wandb_run is not None:
                _log = {k: (v.item() if hasattr(v, 'item') else v) for k, v in metrics.items()}
                _log['train/loss']       = _log.pop('loss', None)
                _log['train/lr']         = _log.pop('lr', None)
                _log['train/gpu_mem_mb'] = _log.pop('gpu_mem_mb', None)
                _wandb_run.log({k: v for k, v in _log.items() if v is not None},
                               step=train_state.step)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 0:
            grid_path = os.path.join(config.sample_dir, f'{train_state.step}.png')
            logging.info(f'Saving 8-clip vorticity grid -> {grid_path}')
            # If model is conditional, use the cond from the current batch's
            # first 8 clips so the grid is genuinely conditional samples.
            cond_for_eval = None
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                cond_for_eval = batch[1][:8].to(device)
            samples = _dpm_sample(8, cond=cond_for_eval)
            utils.DCT3D_samples_to_video_grid(
                samples,
                tokens=dataset.data_shape[0],
                low_freqs=config.dataset.low_freqs,
                block_T=config.dataset.block_T,
                block_H=config.dataset.block_H,
                block_W=config.dataset.block_W,
                reverse_order_3d=reverse_order_3d,
                T=config.dataset.T,
                H=config.dataset.H,
                W=config.dataset.W,
                Y_bound=config.dataset.Y_bound,
                n_rows=8,
                frames_per_row=8,
                path=grid_path,
            )
            if _wandb_run is not None:
                import wandb as _wandb
                _wandb_run.log({'samples/grid': _wandb.Image(grid_path)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        _ckpt_min_step = config.train.get('ckpt_min_step', 0)
        if train_state.step >= _ckpt_min_step and train_state.step % config.train.save_interval == 0:
            logging.info(f'Saving checkpoint at step {train_state.step}')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    if accelerator.is_main_process and device.type == 'cuda':
        peak_mb = _peak_gpu_mb(device)
        logging.info(f'Peak GPU memory (training): {peak_mb:.1f} MB  ({peak_mb/1024:.2f} GB)')
    del metrics
    accelerator.wait_for_everyone()
    if _wandb_run is not None:
        _wandb_run.finish()
    logging.info('all done!')


from absl import flags
from absl import app
from ml_collections import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def main(argv):
    config = FLAGS.config
    config.workdir   = FLAGS.workdir or 'exp_dctdiff_karman'
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
