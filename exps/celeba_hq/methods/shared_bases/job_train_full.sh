#!/bin/bash
#SBATCH --job-name=train_shared_bases_full
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=${DATA_ROOT}/ablation_results/shared_bases/logs/full_%j.out
#SBATCH --error=${DATA_ROOT}/ablation_results/shared_bases/logs/full_%j.err

set -euo pipefail
mkdir -p ${DATA_ROOT}/ablation_results/shared_bases/logs

module --force purge
module load anaconda
source activate video_factor

cd ${REPO_ROOT}/exps/celeba_hq/methods/shared_bases

echo "=== START $(date -Is) ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=== RUN full (48h) ==="

# Auto-resume from latest checkpoint if present
LATEST_CKPT=""
EXP_DIR=$(ls -d ${DATA_ROOT}/ablation_results/shared_bases/[0-9]*-AlphaOnlyDiT 2>/dev/null | sort | tail -1 || true)
if [ -n "${EXP_DIR}" ] && [ -d "${EXP_DIR}/checkpoints" ]; then
  LATEST_CKPT=$(ls "${EXP_DIR}/checkpoints/"*.pt 2>/dev/null | sort | tail -1 || true)
fi
RESUME_ARG=""
[ -n "${LATEST_CKPT}" ] && RESUME_ARG="--resume ${LATEST_CKPT}" && echo "Resuming from ${LATEST_CKPT}"

python -u train.py --num-workers 2 --wandb-run-name shared_bases ${RESUME_ARG}

echo "=== DONE $(date -Is) ==="
