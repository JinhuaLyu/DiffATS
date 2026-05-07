#!/bin/bash
#SBATCH -J burgers2d_viz
#SBATCH -A <ACCOUNT>
#SBATCH -p gpu-debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 0:10:00
#SBATCH -o ${DATA_ROOT}/original_data/burgers_2d/test_data/logs/viz_%j.out
#SBATCH -e ${DATA_ROOT}/original_data/burgers_2d/test_data/logs/viz_%j.err

mkdir -p ${DATA_ROOT}/original_data/burgers_2d/test_data/logs

module load anaconda/2024.02-py311
source activate video_factor

cd ${REPO_ROOT}/tensor_physics/exp_burgers_2d/data_generation

python viz_sample.py
