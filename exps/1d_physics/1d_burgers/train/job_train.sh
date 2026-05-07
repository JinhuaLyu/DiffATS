#!/bin/bash
#SBATCH --job-name=burgers1d_dit
#SBATCH --partition=ghx4
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=${HOME}/factor_diffusion/1d_physics/1d_burgers/train/logs/train_%j.out
#SBATCH --error=${HOME}/factor_diffusion/1d_physics/1d_burgers/train/logs/train_%j.err

set -euo pipefail
mkdir -p ${HOME}/factor_diffusion/1d_physics/1d_burgers/train/logs

echo "[node ] $(hostname)  $(date)"
echo "[gpu  ] $(nvidia-smi -L 2>/dev/null | head)"

module load python/miniforge3_pytorch/2.11.0

cd ${HOME}/factor_diffusion/1d_physics/1d_burgers/train

python -u train_burgers_1d.py \
    --config ${HOME}/factor_diffusion/1d_physics/1d_burgers/configs/train_v1.yaml

echo "[done ] $(date)"
