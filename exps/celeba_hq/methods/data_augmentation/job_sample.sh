#!/bin/bash
#SBATCH --job-name=sample_data_augmentation
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=${DATA_ROOT}/ablation_results/data_augmentation/logs/sample_%j.out
#SBATCH --error=${DATA_ROOT}/ablation_results/data_augmentation/logs/sample_%j.err

set -euo pipefail
mkdir -p ${DATA_ROOT}/ablation_results/data_augmentation/logs
mkdir -p ${DATA_ROOT}/ablation_results/data_augmentation/samples/{images,latents}

module --force purge
module load anaconda
source activate video_factor

cd ${REPO_ROOT}/exps/celeba_hq/methods

echo "=== START $(date -Is) ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=== sampling 10000 from method=data_augmentation ==="

python -u sample_core.py --method data_augmentation --num-samples 10000 --batch-size 32 --sampler ddim --num-sampling-steps 250

echo "=== DONE $(date -Is) ==="
