#!/bin/bash
#SBATCH -J karman_tucker_test
#SBATCH -A <ACCOUNT>
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 0:10:00
#SBATCH -o ${DATA_ROOT}/tucker_factors/karman_vortex_2d/tucker_karman_rT10_rX128_rY30/test_data/logs/save_tucker_%j.out
#SBATCH -e ${DATA_ROOT}/tucker_factors/karman_vortex_2d/tucker_karman_rT10_rX128_rY30/test_data/logs/save_tucker_%j.err

mkdir -p ${DATA_ROOT}/tucker_factors/karman_vortex_2d/tucker_karman_rT10_rX128_rY30/test_data/logs

module load anaconda/2024.02-py311
source activate video_factor

cd ${REPO_ROOT}/tensor_physics/exp_karman_vortex

python -u save_tucker_karman.py \
    --data_dir ${DATA_ROOT}/original_data/karman_vortex_2d/test_data \
    --shard_pattern "test_shard_*.pt" \
    --out_dir   ${DATA_ROOT}/tucker_factors/karman_vortex_2d/tucker_karman_rT10_rX128_rY30/test_data \
    --anchor_path ${DATA_ROOT}/tucker_factors/karman_vortex_2d/tucker_karman_rT10_rX128_rY30/ref_anchor.pt \
    --rank_T 10 --rank_X 128 --rank_Y 30 --r_ic 30 \
    --n_workers $SLURM_CPUS_PER_TASK
