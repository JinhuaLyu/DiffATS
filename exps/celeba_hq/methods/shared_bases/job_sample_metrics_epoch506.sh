#!/bin/bash
#SBATCH --job-name=sm_sb_e506
#SBATCH --account=eng260004-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:30:00
#SBATCH --output=/anvil/projects/x-eng260004/factor_diffusion/ablation_results/shared_bases/logs/sample_metrics_epoch506_%j.out
#SBATCH --error=/anvil/projects/x-eng260004/factor_diffusion/ablation_results/shared_bases/logs/sample_metrics_epoch506_%j.err
#SBATCH --mail-user=jinhualyu2024@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

CKPT=/anvil/projects/x-eng260004/factor_diffusion/ablation_results/shared_bases/004-AlphaOnlyDiT/checkpoints/epoch_00506.pt
SUBDIR=samples_epoch506
SAMPLES_ROOT=/anvil/projects/x-eng260004/factor_diffusion/ablation_results/shared_bases
IMAGES_DIR=${SAMPLES_ROOT}/${SUBDIR}/images
METRICS_JSON=${SAMPLES_ROOT}/metrics_shared_bases_epoch506.json

mkdir -p ${SAMPLES_ROOT}/logs
mkdir -p ${SAMPLES_ROOT}/${SUBDIR}/{images,latents}

module --force purge
module load anaconda
source activate video_factor

python -c "import scipy" 2>/dev/null || pip install --user --quiet scipy

cd /home/x-jlyu5/jinhua/DiffATS/exps/celeba_hq/methods

echo "=== START $(date -Is) ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=== ckpt = ${CKPT} ==="
echo "=== sampling 10000 -> ${IMAGES_DIR} ==="

python -u sample_core.py \
    --method shared_bases \
    --num-samples 10000 \
    --batch-size 32 \
    --sampler ddim \
    --num-sampling-steps 250 \
    --ckpt ${CKPT} \
    --samples-subdir ${SUBDIR}

echo "=== sampling done $(date -Is); computing metrics ==="

python -u compute_metrics_one.py \
    --gen-dir ${IMAGES_DIR} \
    --orig-dir /anvil/projects/x-eng260004/factor_diffusion/original_data/celeba \
    --n-samples 10000 \
    --batch-size 64 \
    --label shared_bases_epoch506 \
    --out-json ${METRICS_JSON}

echo "=== DONE $(date -Is) ==="
