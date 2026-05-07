#!/bin/bash
#SBATCH -J karman_lr5e-3
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
#SBATCH -o /anvil/projects/x-eng260004/factor_diffusion/our_method_results/karman_vortex_2d_lr_sweep/logs/lr5e-3_%j.out
#SBATCH -e /anvil/projects/x-eng260004/factor_diffusion/our_method_results/karman_vortex_2d_lr_sweep/logs/lr5e-3_%j.err

mkdir -p /anvil/projects/x-eng260004/factor_diffusion/our_method_results/karman_vortex_2d_lr_sweep/logs

module load anaconda/2024.02-py311
source activate video_factor

cd /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_karman_vortex/train

python -u train_karman_2d.py \
    --config /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_karman_vortex/configs/train_v1.yaml \
    --lr 5e-3 \
    --n_epochs 500 \
    --outdir /anvil/projects/x-eng260004/factor_diffusion/our_method_results/karman_vortex_2d_lr_sweep/lr_5e-3 \
    --wandb_run lrsweep_karman2d_lr5e-3
