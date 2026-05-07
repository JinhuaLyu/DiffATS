#!/bin/bash
#SBATCH -J burgers2d_gen_ep500
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 1:30:00
#SBATCH -o /anvil/projects/x-eng260004/factor_diffusion/our_method_generation/burgers_2d/logs/gen_ep500_%j.out
#SBATCH -e /anvil/projects/x-eng260004/factor_diffusion/our_method_generation/burgers_2d/logs/gen_ep500_%j.err

mkdir -p /anvil/projects/x-eng260004/factor_diffusion/our_method_generation/burgers_2d/logs

module load anaconda/2024.02-py311
source activate video_factor

cd /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_burgers_2d/generate

# Generate 5 seeds of factors at epoch 500
python -u gen_burgers_2d.py \
    --epoch 500 \
    --output_dir /anvil/projects/x-eng260004/factor_diffusion/our_method_generation/burgers_2d \
    --batch_size 50 \
    --seeds 0 1 2 3 4 \
    --sample_steps 250 \
    --device cuda:0

# Reconstruct seed 0 into (N,200,128,128) video tensor, save as _videos.pt
cd /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics
python -u reconstruct_gen.py --exp burgers --seed 0 --epoch 500
