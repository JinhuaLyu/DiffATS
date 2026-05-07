#!/bin/bash
#SBATCH --job-name=train_our_method_smoke
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=${DATA_ROOT}/ablation_results/our_method/logs/smoke_%j.out
#SBATCH --error=${DATA_ROOT}/ablation_results/our_method/logs/smoke_%j.err

set -euo pipefail
mkdir -p ${DATA_ROOT}/ablation_results/our_method/logs

module --force purge
module load anaconda
source activate video_factor

cd ${REPO_ROOT}/exps/celeba_hq/methods/our_method

echo "=== START $(date -Is) ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=== RUN smoke (1h) ==="

# All training-size hyperparameters come from our_method/train.yaml.
# Only path/wandb overrides here (yaml has stale 'Exp_p32r32_acceleration' paths).
python -u train.py \
    --shard-dir ${DATA_ROOT}/tucker_factors/celeba/our_method \
    --alpha-stats-path ${DATA_ROOT}/tucker_factors/celeba/our_method/alpha_stats_procrustes_refimg_p32_r32.pt \
    --vhat-stats-path  ${DATA_ROOT}/tucker_factors/celeba/our_method/vhat_stats_procrustes_refimg_p32_r32.pt \
    --num-workers 2 \
    --wandb-run-name our_method_smoke

echo "=== DONE $(date -Is) ==="
