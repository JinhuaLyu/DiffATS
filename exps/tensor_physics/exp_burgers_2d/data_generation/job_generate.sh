#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -J burgers2d_gen
#SBATCH -p gengpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o ${DATA_ROOT}/burgers_2d/logs/gen_%j.out
#SBATCH -e ${DATA_ROOT}/burgers_2d/logs/gen_%j.err

mkdir -p ${DATA_ROOT}/burgers_2d/logs

cd /gpfs${REPO_ROOT}/tensor_physics/exp_burgers_2d/data_generation

python generate_data.py
