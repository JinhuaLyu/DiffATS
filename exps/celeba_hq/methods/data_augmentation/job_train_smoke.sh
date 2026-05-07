#!/bin/bash
#SBATCH --job-name=train_data_aug_smoke
#SBATCH --account=eng260004-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/anvil/projects/x-eng260004/factor_diffusion/ablation_results/data_augmentation/logs/smoke_%j.out
#SBATCH --error=/anvil/projects/x-eng260004/factor_diffusion/ablation_results/data_augmentation/logs/smoke_%j.err
#SBATCH --mail-user=jinhualyu2024@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail
mkdir -p /anvil/projects/x-eng260004/factor_diffusion/ablation_results/data_augmentation/logs

module --force purge
module load anaconda
source activate video_factor

cd /home/x-jlyu5/jinhua/DiffATS/exps/celeba_hq/methods/data_augmentation

echo "=== START $(date -Is) ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=== RUN smoke (1h) ==="

python -u train.py --num-workers 2 --wandb-run-name data_augmentation_smoke

echo "=== DONE $(date -Is) ==="
