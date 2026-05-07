#!/bin/bash
#SBATCH --job-name=reaction_1d_gen
#SBATCH --partition=ghx4
#SBATCH --account=bgxp-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/u/jlyu5/factor_diffusion/1d_physics/1d_reaction_diffusion/data_generation/logs/gen_%j.out
#SBATCH --error=/u/jlyu5/factor_diffusion/1d_physics/1d_reaction_diffusion/data_generation/logs/gen_%j.err

set -euo pipefail
mkdir -p /u/jlyu5/factor_diffusion/1d_physics/1d_reaction_diffusion/data_generation/logs

echo "[node ] $(hostname)  $(date)"
echo "[gpu  ] $(nvidia-smi -L 2>/dev/null | head)"

module load python/miniforge3_pytorch/2.11.0

echo "[setup] ensuring jax[cuda12]==0.10.0 in user site-packages"
pip install --user --quiet "jax[cuda12]==0.10.0"
export PYTHONPATH=/u/jlyu5/.local/lib/python3.12/site-packages:${PYTHONPATH:-}

python -c "import jax; print('[setup] jax', jax.__version__, 'devices:', jax.devices())"

cd /u/jlyu5/factor_diffusion/1d_physics/1d_reaction_diffusion/data_generation

OUT_DIR=/work/hdd/bgxp/factor_diffusion/original_data/reaction_1d
mkdir -p "${OUT_DIR}"

# ── Train: 10000 samples, master seed 2022 ──────────────────────────────────
echo
echo "[==train==] generating 10000 samples (seed=2022)"
python generate_dataset.py \
    --n_samples 10000 \
    --batch_size 200 \
    --seed 2022 \
    --out_path "${OUT_DIR}/reaction_1d_train.pt"

# ── Test: 500 samples, independent master seed 20260426 ─────────────────────
echo
echo "[==test==] generating 500 samples (seed=20260426)"
python generate_dataset.py \
    --n_samples 500 \
    --batch_size 200 \
    --seed 20260426 \
    --out_path "${OUT_DIR}/reaction_1d_test.pt"

echo "[done ] $(date)"
