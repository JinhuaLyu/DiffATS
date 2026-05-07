#!/bin/bash
#SBATCH --account=bgxp-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --job-name=svd_ablation_n100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/x-jlyu5/jinhua/DiffATS/exps/celeba_hq/ablation/slurm_%j.out
#SBATCH --error=/home/x-jlyu5/jinhua/DiffATS/exps/celeba_hq/ablation/slurm_%j.err

set -euo pipefail
module load python/3.11.9
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export N_IMAGES=100

cd /home/x-jlyu5/jinhua/DiffATS/exps/celeba_hq/ablation
echo "=== START $(date -Is) host=$(hostname) job=${SLURM_JOB_ID} ==="
python3 -u svd_patchify_ablation.py
echo "=== DONE $(date -Is) ==="
