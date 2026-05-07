#!/bin/bash
#SBATCH --job-name=reaction1d_gen_ep1000
#SBATCH --partition=ghx4
#SBATCH --account=bgxp-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/u/jlyu5/factor_diffusion/1d_physics/1d_reaction_diffusion/generate/logs/gen_ep1000_%j.out
#SBATCH --error=/u/jlyu5/factor_diffusion/1d_physics/1d_reaction_diffusion/generate/logs/gen_ep1000_%j.err

set -euo pipefail
mkdir -p /u/jlyu5/factor_diffusion/1d_physics/1d_reaction_diffusion/generate/logs

echo "[node ] $(hostname)  $(date)"
echo "[gpu  ] $(nvidia-smi -L 2>/dev/null | head)"

module load python/miniforge3_pytorch/2.11.0

cd /u/jlyu5/factor_diffusion/1d_physics/1d_reaction_diffusion/generate

OUTPUT_DIR=/work/hdd/bgxp/factor_diffusion/our_method_generation/reaction_1d
mkdir -p "${OUTPUT_DIR}"

echo
echo "[==gen==] generating 5 seeds at epoch=1000"
python -u gen_reaction_1d.py \
    --epoch 1000 \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 50 \
    --seeds 0 1 2 3 4 \
    --sample_steps 250 \
    --device cuda:0

echo
echo "[==metrics==] computing L1 / L2 / RMSE vs original physical GT"
python -u metrics_reaction_1d.py \
    --gen_dir "${OUTPUT_DIR}" \
    --epoch 1000 \
    --seeds 0 1 2 3 4

echo "[done ] $(date)"
