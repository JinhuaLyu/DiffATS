#!/bin/bash
#SBATCH --job-name=burgers1d_factor_cpu
#SBATCH --partition=ghx4
#SBATCH --account=bgxp-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/u/jlyu5/factor_diffusion/1d_physics/1d_burgers/data_tucker/logs/factor_cpu_%j.out
#SBATCH --error=/u/jlyu5/factor_diffusion/1d_physics/1d_burgers/data_tucker/logs/factor_cpu_%j.err

set -euo pipefail
mkdir -p /u/jlyu5/factor_diffusion/1d_physics/1d_burgers/data_tucker/logs

echo "[node ] $(hostname)  $(date)"

module load python/miniforge3_pytorch/2.11.0

cd /u/jlyu5/factor_diffusion/1d_physics/1d_burgers/data_tucker

TRAIN_PATH=/work/hdd/bgxp/factor_diffusion/original_data/burgers_1d/burgers_1d.pt
TEST_PATH=/work/hdd/bgxp/factor_diffusion/original_data/burgers_1d/burgers_1d_test.pt
OUT_DIR=/work/hdd/bgxp/factor_diffusion/tucker_factors/burgers_1d

mkdir -p "${OUT_DIR}"

# Hide any GPUs that might be allocated, force CPU path.
export CUDA_VISIBLE_DEVICES=""

python save_factors_burgers_1d.py \
    --train_path "${TRAIN_PATH}" \
    --test_path  "${TEST_PATH}" \
    --out_dir    "${OUT_DIR}" \
    --rank 32 \
    --ref_seed 42 \
    --batch_size 50 \
    --device cpu

echo "[done ] $(date)"
