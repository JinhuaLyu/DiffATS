#!/bin/bash
#SBATCH -J burgers2d_gen
#SBATCH -A <ACCOUNT>
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o ${DATA_ROOT}/our_method_generation/burgers_2d/logs/gen_%j.out
#SBATCH -e ${DATA_ROOT}/our_method_generation/burgers_2d/logs/gen_%j.err

mkdir -p ${DATA_ROOT}/our_method_generation/burgers_2d/logs

module load anaconda/2024.02-py311
source activate video_factor

cd ${REPO_ROOT}/tensor_physics/exp_burgers_2d/generate

python -u gen_burgers_2d.py \
    --epoch 200 \
    --output_dir ${DATA_ROOT}/our_method_generation/burgers_2d \
    --batch_size 50 \
    --seeds 0 1 2 3 4 \
    --sample_steps 250 \
    --device cuda:0
