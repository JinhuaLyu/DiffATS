#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --account=eng260004-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/anvil/projects/x-eng260004/factor_diffusion/ablation_results/logs/metrics_%j.out
#SBATCH --error=/anvil/projects/x-eng260004/factor_diffusion/ablation_results/logs/metrics_%j.err
#SBATCH --mail-user=jinhualyu2024@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

ABL=/anvil/projects/x-eng260004/factor_diffusion/ablation_results
ORIG=/anvil/projects/x-eng260004/factor_diffusion/original_data/celeba
mkdir -p "${ABL}/logs"

module --force purge
module load anaconda
source activate video_factor

# Resolve celeba_hq root from this script's location
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELEBA_HQ="$(cd "${HERE}/../.." && pwd)"
cd "${CELEBA_HQ}"

echo "=== START $(date -Is) ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u metrics/compute_metrics.py \
    --orig-dir "${ORIG}" \
    --n-samples 10000 \
    --batch-size 64 \
    --out-json "${ABL}/metrics_results.json" \
    --gen-spec \
        our_method=${ABL}/our_method/samples/images \
        no_alignment=${ABL}/no_alignment/samples/images \
        shared_bases=${ABL}/shared_bases/samples/images \
        data_augmentation=${ABL}/data_augmentation/samples/images

echo "=== DONE $(date -Is) ==="
