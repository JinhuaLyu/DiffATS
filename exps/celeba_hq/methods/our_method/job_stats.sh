#!/bin/bash
#SBATCH --job-name=celeba_p32r32_stats
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=${DATA_ROOT}/tucker_factors/celeba/our_method/logs/stats_%j.out
#SBATCH --error=${DATA_ROOT}/tucker_factors/celeba/our_method/logs/stats_%j.err

set -euo pipefail

mkdir -p ${DATA_ROOT}/tucker_factors/celeba/our_method/logs

module --force purge
module load anaconda
source activate video_factor

cd ${REPO_ROOT}/exps/celeba_hq/methods/our_method

echo "=== START $(date -Is) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "=== alpha stats (per-rank) ==="
python -u compute_alpha_stats_refimg.py
echo "=== vhat stats (scalar) ==="
python -u compute_vhat_stats_refimg.py
echo "=== DONE $(date -Is) ==="
