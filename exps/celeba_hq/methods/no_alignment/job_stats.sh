#!/bin/bash
#SBATCH --job-name=celeba_no_align_stats
#SBATCH --account=eng260004-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/celeba/no_alignment/logs/stats_%j.out
#SBATCH --error=/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/celeba/no_alignment/logs/stats_%j.err

set -euo pipefail

mkdir -p /anvil/projects/x-eng260004/factor_diffusion/tucker_factors/celeba/no_alignment/logs

module --force purge
module load anaconda
source activate video_factor

cd /home/x-jlyu5/jinhua/DiffATS/exps/celeba_hq/methods/no_alignment

echo "=== START $(date -Is) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "=== alpha stats (per-rank) ==="
python -u compute_alpha_stats_no_alignment.py
echo "=== vhat stats (scalar) ==="
python -u compute_vhat_stats_no_alignment.py
echo "=== DONE $(date -Is) ==="
