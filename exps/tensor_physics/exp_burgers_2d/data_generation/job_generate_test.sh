#!/bin/bash
#SBATCH -J burgers2d_test_gen
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o /anvil/projects/x-eng260004/factor_diffusion/original_data/burgers_2d/test_data/logs/gen_%j.out
#SBATCH -e /anvil/projects/x-eng260004/factor_diffusion/original_data/burgers_2d/test_data/logs/gen_%j.err

mkdir -p /anvil/projects/x-eng260004/factor_diffusion/original_data/burgers_2d/test_data/logs

module load anaconda/2024.02-py311
source activate video_factor

cd /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_burgers_2d/data_generation

python generate_test_data.py
