#!/bin/bash
#SBATCH -J karman2d_tucker_dit
#SBATCH -A <ACCOUNT>
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 5:00:00
#SBATCH -o ${DATA_ROOT}/our_method_results/karman_vortex_2d/logs/train_%j.out
#SBATCH -e ${DATA_ROOT}/our_method_results/karman_vortex_2d/logs/train_%j.err

mkdir -p ${DATA_ROOT}/our_method_results/karman_vortex_2d/logs

module load anaconda/2024.02-py311
source activate video_factor

cd ${REPO_ROOT}/tensor_physics/exp_karman_vortex/train

python -u train_karman_2d.py \
    --config ${REPO_ROOT}/tensor_physics/exp_karman_vortex/configs/train_v1.yaml
