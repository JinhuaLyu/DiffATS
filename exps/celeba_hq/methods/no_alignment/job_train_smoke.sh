#!/bin/bash
#SBATCH --job-name=train_no_alignment_smoke
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=${DATA_ROOT}/ablation_results/no_alignment/logs/smoke_%j.out
#SBATCH --error=${DATA_ROOT}/ablation_results/no_alignment/logs/smoke_%j.err

set -euo pipefail
mkdir -p ${DATA_ROOT}/ablation_results/no_alignment/logs

module --force purge
module load anaconda
source activate video_factor

cd ${REPO_ROOT}/exps/celeba_hq/methods/no_alignment

echo "=== START $(date -Is) ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=== RUN smoke (1h) ==="

python -u train.py --num-workers 2 --wandb-run-name no_alignment_smoke

echo "=== DONE $(date -Is) ==="
