#!/bin/bash
#SBATCH -J karman2d_gen
#SBATCH -A <ACCOUNT>
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 1:30:00
#SBATCH -o ${DATA_ROOT}/our_method_generation/karman_vortex_2d/logs/gen_%j.out
#SBATCH -e ${DATA_ROOT}/our_method_generation/karman_vortex_2d/logs/gen_%j.err

mkdir -p ${DATA_ROOT}/our_method_generation/karman_vortex_2d/logs

module load anaconda/2024.02-py311
source activate video_factor

cd ${REPO_ROOT}/tensor_physics/exp_karman_vortex/generate

python -u gen_karman_2d.py \
    --ckpt ${DATA_ROOT}/our_method_results/karman_vortex_2d/checkpoints/epoch00200_step0062400.pt \
    --output_dir ${DATA_ROOT}/our_method_generation/karman_vortex_2d \
    --batch_size 50 \
    --seeds 0 1 2 3 4 \
    --sample_steps 250 \
    --device cuda:0
