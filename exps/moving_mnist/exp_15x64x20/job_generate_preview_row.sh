#!/bin/bash
#SBATCH -J mm_preview_row
#SBATCH -A <ACCOUNT>
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 00:10:00
#SBATCH -o ${DATA_ROOT}/our_method_generation/moving_mnist/logs/preview_row_%j.out
#SBATCH -e ${DATA_ROOT}/our_method_generation/moving_mnist/logs/preview_row_%j.err

# Generate a row of preview frames from a single seed.
#
# Override via env vars:
#   CKPT          : checkpoint path
#   OUTDIR        : where to write PNGs
#   SEED          : sampling seed              default: 0
#   N_VIDEOS      : number of videos in row    default: 10
#   BATCH         : batch size                 default: 10
#   SAMPLE_STEPS  : diffusion steps            default: 1000
#   N_FRAMES      : frames per video           default: 5
#
# Example: sweep seeds via SEED=$i sbatch job_generate_preview_row.sh

set -euo pipefail

: "${CKPT:=${DATA_ROOT}/our_method_results/moving_mnist/exp_15x64x20_output/checkpoints/epoch2000.pt}"
: "${OUTDIR:=${DATA_ROOT}/our_method_generation/moving_mnist/preview_rows}"
: "${SEED:=0}"
: "${N_VIDEOS:=10}"
: "${BATCH:=10}"
: "${SAMPLE_STEPS:=1000}"
: "${N_FRAMES:=5}"

mkdir -p "${OUTDIR}"
mkdir -p ${DATA_ROOT}/our_method_generation/moving_mnist/logs

module load anaconda/2024.02-py311
source activate video_factor
export JAX_PLATFORMS=cpu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

echo "[CFG] CKPT=${CKPT}  OUTDIR=${OUTDIR}  SEED=${SEED}"

python -u generate_preview_row.py \
    --ckpt "${CKPT}" \
    --outdir "${OUTDIR}" \
    --seed "${SEED}" \
    --n_videos "${N_VIDEOS}" --batch_size "${BATCH}" \
    --sample_steps "${SAMPLE_STEPS}" --n_frames "${N_FRAMES}"
