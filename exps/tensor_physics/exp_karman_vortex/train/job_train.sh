#!/bin/bash
#SBATCH -J karman2d_tucker_dit
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 5:00:00
#SBATCH -o /anvil/projects/x-eng260004/factor_diffusion/our_method_results/karman_vortex_2d/logs/train_%j.out
#SBATCH -e /anvil/projects/x-eng260004/factor_diffusion/our_method_results/karman_vortex_2d/logs/train_%j.err

mkdir -p /anvil/projects/x-eng260004/factor_diffusion/our_method_results/karman_vortex_2d/logs

module load anaconda/2024.02-py311
source activate video_factor

cd /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_karman_vortex/train

python -u train_karman_2d.py \
    --config /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_karman_vortex/configs/train_v1.yaml
