#!/bin/bash
# ============================================================
# FID Sampling -- Procrustes RefImg JointDiT  [ACCELERATION VER]
#
# Key changes vs original:
#   1. Async PNG saving  (ThreadPoolExecutor, compress_level=1)
#      -> saves overlap with GPU compute; ~3-5x faster I/O
#   2. Dropped channels_last on latent (wrong for H>>W tensors)
#   3. Optional --compile flag (torch.compile, ~20-30% GPU gain)
#   4. batch-size raised to 256
# ============================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

BASE_DIR="/projects/p33154/factor_diff_outputs/Exp_CelebA_HQ_p32r16_acceleration"

RESULTS_DIR="${BASE_DIR}/train_outputs"
OUT_DIR="${BASE_DIR}/generated_images_fid_10k_DDIM"

PY_SCRIPT="/projects/p33154/factor_diff_Exp/CelebA_HQ_1024_Exp_p32r16/sample_procrustes_fid_resume_DDIM_acceleration.py"

python "${PY_SCRIPT}" \
    --hidden-size 768 \
    --depth 12 \
    --num-heads 12 \
    --mlp-ratio 4.0 \
    \
    --results-dir "${RESULTS_DIR}" \
    --use-ema \
    \
    --alpha-stats-path "${BASE_DIR}/alpha_stats_procrustes_refimg_p32_r16.pt" \
    --vhat-stats-path  "${BASE_DIR}/vhat_stats_procrustes_refimg_p32_r16.pt" \
    --ref-anchor-path  "${BASE_DIR}/celebahq1024_patchsvd_procrustes_refimg_p32_r16/ref_anchor.pt" \
    --norm-std 0.5 \
    \
    --img-hw 1024 \
    --patch 32 \
    --svd-rank 16 \
    \
    --output-dir "${OUT_DIR}" \
    --num-images 10000 \
    --batch-size 256 \
    --sampler ddim \
    --num-sampling-steps 250 \
    --ddim-eta 0.0 \
    --seed 42
    # --compile      <- uncomment if PyTorch >= 2.0 and first run is acceptable (~5min warmup)
