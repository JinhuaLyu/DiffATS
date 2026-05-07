#!/bin/bash
#SBATCH -J burgers_lr5e-4
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 05:00:00
#SBATCH -o /anvil/projects/x-eng260004/factor_diffusion/our_method_results/burgers_2d_lr_sweep/logs/lr5e-4_%j.out
#SBATCH -e /anvil/projects/x-eng260004/factor_diffusion/our_method_results/burgers_2d_lr_sweep/logs/lr5e-4_%j.err

mkdir -p /anvil/projects/x-eng260004/factor_diffusion/our_method_results/burgers_2d_lr_sweep/logs

module load anaconda/2024.02-py311
source activate video_factor

cd /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_burgers_2d/train

python -u train_burgers_2d.py \
    --config /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_burgers_2d/configs/train_v1.yaml \
    --lr 5e-4 \
    --n_epochs 500 \
    --outdir /anvil/projects/x-eng260004/factor_diffusion/our_method_results/burgers_2d_lr_sweep/lr_5e-4 \
    --wandb_run lrsweep_burgers2d_lr5e-4
