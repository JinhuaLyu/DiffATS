#!/bin/bash
#SBATCH --job-name=train_no_alignment_full
#SBATCH --account=eng260004-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/anvil/projects/x-eng260004/factor_diffusion/ablation_results/no_alignment/logs/full_%j.out
#SBATCH --error=/anvil/projects/x-eng260004/factor_diffusion/ablation_results/no_alignment/logs/full_%j.err
#SBATCH --mail-user=jinhualyu2024@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail
mkdir -p /anvil/projects/x-eng260004/factor_diffusion/ablation_results/no_alignment/logs

module --force purge
module load anaconda
source activate video_factor

cd /home/x-jlyu5/jinhua/DiffATS/exps/celeba_hq/methods/no_alignment

echo "=== START $(date -Is) ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=== RUN full (48h) ==="

LATEST_CKPT=""
EXP_DIR=$(ls -d /anvil/projects/x-eng260004/factor_diffusion/ablation_results/no_alignment/[0-9]*-JointDiT 2>/dev/null | sort | tail -1 || true)
if [ -n "${EXP_DIR}" ] && [ -d "${EXP_DIR}/checkpoints" ]; then
  LATEST_CKPT=$(ls "${EXP_DIR}/checkpoints/"*.pt 2>/dev/null | sort | tail -1 || true)
fi
RESUME_ARG=""
[ -n "${LATEST_CKPT}" ] && RESUME_ARG="--resume ${LATEST_CKPT}" && echo "Resuming from ${LATEST_CKPT}"

python -u train.py --num-workers 2 --wandb-run-name no_alignment ${RESUME_ARG}

echo "=== DONE $(date -Is) ==="
