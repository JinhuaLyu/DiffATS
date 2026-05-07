#!/bin/bash
#SBATCH -J burgers2d_tucker_test
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o /anvil/projects/x-eng260004/factor_diffusion/tucker_factors/burgers_2d/tucker_burgers_rT5_rH20_rW20/test_data/logs/save_tucker_%j.out
#SBATCH -e /anvil/projects/x-eng260004/factor_diffusion/tucker_factors/burgers_2d/tucker_burgers_rT5_rH20_rW20/test_data/logs/save_tucker_%j.err

mkdir -p /anvil/projects/x-eng260004/factor_diffusion/tucker_factors/burgers_2d/tucker_burgers_rT5_rH20_rW20/test_data/logs

module load anaconda/2024.02-py311
source activate video_factor

cd /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_burgers_2d/data_tucker

python -u save_tucker_burgers.py \
    --data_dir /anvil/projects/x-eng260004/factor_diffusion/original_data/burgers_2d/test_data \
    --shard_pattern "test_shard_*.pt" \
    --out_dir   /anvil/projects/x-eng260004/factor_diffusion/tucker_factors/burgers_2d/tucker_burgers_rT5_rH20_rW20/test_data \
    --anchor_path /anvil/projects/x-eng260004/factor_diffusion/tucker_factors/burgers_2d/tucker_burgers_rT5_rH20_rW20/ref_anchor.pt \
    --rank_T 5 --rank_H 20 --rank_W 20 --r_ic 20 \
    --n_workers $SLURM_CPUS_PER_TASK
