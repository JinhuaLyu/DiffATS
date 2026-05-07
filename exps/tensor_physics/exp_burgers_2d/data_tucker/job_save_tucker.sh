#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -J burgers2d_tucker
#SBATCH -p normal
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o ${OUTPUT_ROOT}/burgers_2d/tucker_factors/logs/save_tucker_%j.out
#SBATCH -e ${OUTPUT_ROOT}/burgers_2d/tucker_factors/logs/save_tucker_%j.err

mkdir -p ${OUTPUT_ROOT}/burgers_2d/tucker_factors/logs

cd ${REPO_ROOT}/tensor_physics/exp_burgers_2d/data_tucker

python save_tucker_burgers.py \
    --data_dir ${DATA_ROOT}/burgers_2d \
    --out_dir  ${OUTPUT_ROOT}/burgers_2d/tucker_factors/tucker_burgers_rT5_rH20_rW20 \
    --rank_T 5 --rank_H 20 --rank_W 20 --r_ic 20 \
    --n_workers $SLURM_CPUS_PER_TASK
