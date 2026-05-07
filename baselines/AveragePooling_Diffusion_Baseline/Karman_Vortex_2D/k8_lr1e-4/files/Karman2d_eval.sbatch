#!/bin/bash
#SBATCH --job-name=Karman_k8_fixlr_eval
#SBATCH --account=bgxp-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/bgxp/ezhou1/factor_diffusion_proj/Experiments_Output/Karman_Vortex_2D/k8_H200_fixlr/logs/eval_%j.out
#SBATCH --error=/scratch/bgxp/ezhou1/factor_diffusion_proj/Experiments_Output/Karman_Vortex_2D/k8_H200_fixlr/logs/eval_%j.err

set -euo pipefail

export PATH="/u/ezhou1/.conda/envs/CelebA_cu124/bin:$PATH"

OUTPUT_DIR=/scratch/bgxp/ezhou1/factor_diffusion_proj/Experiments_Output/Karman_Vortex_2D/k8_H200_fixlr

mkdir -p ${OUTPUT_DIR}/logs

cd /scratch/bgxp/ezhou1/factor_diffusion_proj/Experiments/Karman_Vortex_2d

python /scratch/bgxp/ezhou1/factor_diffusion_proj/Experiments/Karman_Vortex_2d/Karman_eval.py \
    --test_dir          /scratch/bgxp/ezhou1/factor_diffusion_proj/data/karman_vortex_2d/test_data \
    --pool_k            8 \
    --orig_size         128 \
    --num_test_samples  500 \
    --samples_per_shard 50 \
    --num_seeds         5 \
    --ckpt_path         ${OUTPUT_DIR}/checkpoints/final.pt \
    --hidden_dim        512 \
    --num_layers        12 \
    --num_heads         8 \
    --diff_timesteps    1000 \
    --ddim_steps        250 \
    --output_dir        ${OUTPUT_DIR}
