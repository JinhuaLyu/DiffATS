#!/bin/bash
#SBATCH --job-name=celeba_p32r32_preprocess
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=${DATA_ROOT}/tucker_factors/celeba/our_method/logs/preprocess_%j.out
#SBATCH --error=${DATA_ROOT}/tucker_factors/celeba/our_method/logs/preprocess_%j.err

set -euo pipefail

mkdir -p ${DATA_ROOT}/tucker_factors/celeba/our_method/logs

module --force purge
module load anaconda
source activate video_factor

cd ${REPO_ROOT}/exps/celeba_hq/methods/our_method

echo "=== START $(date -Is) ==="
echo "Host:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "GPUs:    ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda?', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "=== RUN ==="

python -u all_save_procrustes_svd_refimg_acceleration.py

echo "=== DONE $(date -Is) ==="
