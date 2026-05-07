#!/bin/bash
#SBATCH -J burgers2d_to1500
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 05:00:00
#SBATCH -o /anvil/projects/x-eng260004/factor_diffusion/our_method_results/burgers_2d/logs/train_%j.out
#SBATCH -e /anvil/projects/x-eng260004/factor_diffusion/our_method_results/burgers_2d/logs/train_%j.err

mkdir -p /anvil/projects/x-eng260004/factor_diffusion/our_method_results/burgers_2d/logs

module load anaconda/2024.02-py311
source activate video_factor

cd /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_burgers_2d/train

# Auto-resume from latest ckpt (epoch 1000), continue until epoch 1500
python -u train_burgers_2d.py \
    --config /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_burgers_2d/configs/train_v1.yaml \
    --n_epochs 1500
