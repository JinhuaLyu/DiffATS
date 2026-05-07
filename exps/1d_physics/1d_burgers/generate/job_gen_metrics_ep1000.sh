#!/bin/bash
#SBATCH --job-name=burgers1d_gen_ep1000
#SBATCH --partition=ghx4
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=${HOME}/factor_diffusion/1d_physics/1d_burgers/generate/logs/gen_ep1000_%j.out
#SBATCH --error=${HOME}/factor_diffusion/1d_physics/1d_burgers/generate/logs/gen_ep1000_%j.err

set -euo pipefail
mkdir -p ${HOME}/factor_diffusion/1d_physics/1d_burgers/generate/logs

echo "[node ] $(hostname)  $(date)"
echo "[gpu  ] $(nvidia-smi -L 2>/dev/null | head)"

module load python/miniforge3_pytorch/2.11.0

cd ${HOME}/factor_diffusion/1d_physics/1d_burgers/generate

OUTPUT_DIR=${DATA_ROOT}/our_method_generation/burgers_1d
mkdir -p "${OUTPUT_DIR}"

echo
echo "[==gen==] generating 5 seeds at epoch=1000"
python -u gen_burgers_1d.py \
    --epoch 1000 \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 50 \
    --seeds 0 1 2 3 4 \
    --sample_steps 250 \
    --device cuda:0

echo
echo "[==metrics==] computing L1 / L2 / RMSE vs original physical GT"
python -u metrics_burgers_1d.py \
    --gen_dir "${OUTPUT_DIR}" \
    --epoch 1000 \
    --seeds 0 1 2 3 4

echo "[done ] $(date)"
