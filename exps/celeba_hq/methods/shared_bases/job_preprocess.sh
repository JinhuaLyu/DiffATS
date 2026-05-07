#!/bin/bash
#SBATCH --job-name=celeba_global_pca_preprocess
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=${DATA_ROOT}/tucker_factors/celeba/shared_bases/logs/preprocess_%j.out
#SBATCH --error=${DATA_ROOT}/tucker_factors/celeba/shared_bases/logs/preprocess_%j.err

set -euo pipefail

mkdir -p ${DATA_ROOT}/tucker_factors/celeba/shared_bases/logs

module --force purge
module load anaconda
source activate video_factor

cd ${REPO_ROOT}/exps/celeba_hq/methods/shared_bases

echo "=== START $(date -Is) ==="
echo "Host:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "GPUs:    ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda?', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "=== RUN ==="

python -u all_save_global_pca.py

echo "=== DONE $(date -Is) ==="
