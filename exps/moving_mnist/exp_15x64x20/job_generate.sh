#!/bin/bash
#SBATCH -J mm_generate
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o /anvil/projects/x-eng260004/factor_diffusion/our_method_generation/moving_mnist/logs/generate_%j.out
#SBATCH -e /anvil/projects/x-eng260004/factor_diffusion/our_method_generation/moving_mnist/logs/generate_%j.err

# Unified driver for generate.py.
#
# Override any of these via env vars before sbatch (or `sbatch --export=...`):
#   CKPT      : path to checkpoint        (default: epoch2000.pt under our_method_results)
#   OUTDIR    : output directory          (default: our_method_generation/moving_mnist)
#   MODE      : "pt" (full batch tensors) | "gif" (preview gifs)   default: pt
#   N_VIDEOS  : number of videos          default: 10000 (pt) / 10 (gif)
#   BATCH     : batch size                default: 128 (pt) / 10 (gif)
#   SEED      : sampling seed             default: 0
#   SCALE     : classifier-free scale     default: 2.5
#   LOW_CUT   : low-cut threshold         default: 0.1
#
# Examples:
#   sbatch job_generate.sh                                # full 10k tensors, lc=0.1, scale=2.5
#   MODE=gif N_VIDEOS=10 SCALE=5.0 sbatch job_generate.sh # 10 preview gifs, scale=5
#   CKPT=/path/to/lr5e-4/epoch2000.pt OUTDIR=/path/to/lr5e-4 sbatch job_generate.sh

set -euo pipefail

: "${CKPT:=/anvil/projects/x-eng260004/factor_diffusion/our_method_results/moving_mnist/exp_15x64x20_output/checkpoints/epoch2000.pt}"
: "${OUTDIR:=/anvil/projects/x-eng260004/factor_diffusion/our_method_generation/moving_mnist}"
: "${MODE:=pt}"
: "${SEED:=0}"
: "${SCALE:=2.5}"
: "${LOW_CUT:=0.1}"

if [ "$MODE" = "gif" ]; then
    : "${N_VIDEOS:=10}"
    : "${BATCH:=10}"
else
    : "${N_VIDEOS:=10000}"
    : "${BATCH:=128}"
fi

mkdir -p "${OUTDIR}/logs"

module load anaconda/2024.02-py311
source activate video_factor
export JAX_PLATFORMS=cpu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

echo "[CFG] CKPT=${CKPT}"
echo "[CFG] OUTDIR=${OUTDIR}  MODE=${MODE}"
echo "[CFG] N_VIDEOS=${N_VIDEOS}  BATCH=${BATCH}  SEED=${SEED}"
echo "[CFG] SCALE=${SCALE}  LOW_CUT=${LOW_CUT}"

python -u generate.py \
    --ckpt "${CKPT}" \
    --outdir "${OUTDIR}" \
    --mode "${MODE}" \
    --n_videos "${N_VIDEOS}" --batch_size "${BATCH}" \
    --seed "${SEED}" --scale "${SCALE}" --low_cut "${LOW_CUT}"
