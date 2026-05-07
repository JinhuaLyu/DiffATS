#!/bin/bash
#SBATCH -J mm_fvd
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o /anvil/projects/x-eng260004/factor_diffusion/our_method_generation/moving_mnist/logs/fvd_%j.out
#SBATCH -e /anvil/projects/x-eng260004/factor_diffusion/our_method_generation/moving_mnist/logs/fvd_%j.err

# 3-way FVD: real-vs-tucker, tucker-vs-gen, real-vs-gen.
#
# Override via env vars:
#   REAL    : real videos   .pt  default: original_data/moving_mnist_20k_2slow.pt
#   RECON   : tucker recon  .pt  default: moving_mnist_tucker_recon_raw.pt
#   GEN     : generated     .pt  default: moving_mnist_gen_epoch2000_raw_videos.pt
#   OUTDIR  : where JSON results go
#   N       : sample count        default: 10000
#   BATCH   : feature batch       default: 16
#   FACTORS : if set, run reconstruct_tucker.py first to produce GEN from FACTORS
#
# Example for lr5e-4 variant (recon from factors first):
#   FACTORS=/.../lr5e-4/moving_mnist_gen_epoch2000_factors.pt \
#     GEN=/.../lr5e-4/moving_mnist_gen_epoch2000_raw_videos.pt \
#     OUTDIR=/.../lr5e-4 \
#     sbatch job_fvd.sh

set -euo pipefail

ROOT=/anvil/projects/x-eng260004/factor_diffusion
: "${REAL:=${ROOT}/original_data/moving_mnist/moving_mnist_20k_2slow.pt}"
: "${RECON:=${ROOT}/our_method_generation/moving_mnist/moving_mnist_tucker_recon_raw.pt}"
: "${GEN:=${ROOT}/our_method_generation/moving_mnist/moving_mnist_gen_epoch2000_raw_videos.pt}"
: "${OUTDIR:=${ROOT}/our_method_generation/moving_mnist}"
: "${N:=10000}"
: "${BATCH:=16}"
: "${FACTORS:=}"

mkdir -p "${OUTDIR}/logs"

module load anaconda/2024.02-py311
source activate video_factor
export JAX_PLATFORMS=cpu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

echo "[CFG] REAL=${REAL}"
echo "[CFG] RECON=${RECON}"
echo "[CFG] GEN=${GEN}"
echo "[CFG] OUTDIR=${OUTDIR}  N=${N}  BATCH=${BATCH}"

if [ -n "${FACTORS}" ]; then
    echo "=========================================="
    echo "Step 0: reconstruct_tucker  factors -> raw"
    echo "=========================================="
    python -u reconstruct_tucker.py --factors_pt "${FACTORS}" --out "${GEN}"
fi

run_fvd () {
    local name="$1" a="$2" b="$3" out="$4"
    echo ""
    echo "=========================================="
    echo "FVD: ${name}"
    echo "=========================================="
    python -u compute_fvd.py \
        --real "${a}" --gen "${b}" \
        --n "${N}" --batch "${BATCH}" --real_select first \
        --out "${out}"
}

run_fvd "real vs tucker_recon"   "${REAL}"  "${RECON}" "${OUTDIR}/fvd_real_vs_tucker_recon.json"
run_fvd "tucker_recon vs gen"    "${RECON}" "${GEN}"   "${OUTDIR}/fvd_tucker_recon_vs_gen.json"
run_fvd "real vs gen"            "${REAL}"  "${GEN}"   "${OUTDIR}/fvd_real_vs_gen.json"
