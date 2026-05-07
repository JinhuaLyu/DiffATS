import json
import os
import builtins
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


_DEFAULT_STATS_PATH = Path('/scratch/bkx8728/burgers_dctdiff_runs/burgers_stats_3d_b5.json')


def _load_stats(config):
    path = Path(config.dataset.get('stats_path', str(_DEFAULT_STATS_PATH)))
    if not path.exists():
        raise FileNotFoundError(f'{path} not found -- run the matching statis script first')
    stats = json.loads(path.read_text())
    config.dataset.Y_bound = [stats['Y_bound']]
    config.dataset.vor_std  = stats['vor_std']
    logging.info(
        f"Loaded stats: Y_bound={stats['Y_bound']:.6f}, vor_std[:5]={stats['vor_std'][:5]}"
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

    n_cond = int(config.dataset.n_cond_tokens)
    n_total = int(dataset.data_shape[0])
    n_pred = n_total - n_cond
    logging.info(
        f't0-conditional training: total tokens={n_total}, n_cond={n_cond} (t=0 2D-DCT), n_pred={n_pred} (spatiotemporal)'
    )
    assert n_cond > 0 and n_pred > 0

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader
    )
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

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

    score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale))

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        loss = sde.LSimple_cond(
            score_model, _batch, n_cond=n_cond,
            pred=config.pred, reweight=reweight_by_std,
        )
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

    logging.info(f'Start fitting (T0-CONDITIONAL), step={train_state.step}, mixed_precision={config.mixed_precision}')
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            log_dict = dict(step=train_state.step, **metrics)
            logging.info(utils.dct2str(log_dict))
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
    config.workdir   = FLAGS.workdir or '/scratch/bkx8728/burgers_dctdiff_runs/exp_t0cond_default'
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
