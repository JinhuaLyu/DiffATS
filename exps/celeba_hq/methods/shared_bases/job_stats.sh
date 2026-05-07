#!/bin/bash
#SBATCH --job-name=celeba_global_pca_stats
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=${DATA_ROOT}/tucker_factors/celeba/shared_bases/logs/stats_%j.out
#SBATCH --error=${DATA_ROOT}/tucker_factors/celeba/shared_bases/logs/stats_%j.err

set -euo pipefail

mkdir -p ${DATA_ROOT}/tucker_factors/celeba/shared_bases/logs

module --force purge
module load anaconda
source activate video_factor

cd ${REPO_ROOT}/exps/celeba_hq/methods/shared_bases

echo "=== START $(date -Is) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "=== alpha stats (per-rank) ==="
python -u compute_alpha_stats_global.py
echo "=== DONE $(date -Is) ==="
