#!/bin/bash
#SBATCH --job-name=burgers_1d_test_gen
#SBATCH --partition=ghx4
#SBATCH --account=bgxp-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/u/jlyu5/factor_diffusion/1d_physics/1d_burgers/data_generation/logs/gen_test_%j.out
#SBATCH --error=/u/jlyu5/factor_diffusion/1d_physics/1d_burgers/data_generation/logs/gen_test_%j.err

set -euo pipefail
mkdir -p /u/jlyu5/factor_diffusion/1d_physics/1d_burgers/data_generation/logs

echo "[node ] $(hostname)  $(date)"
echo "[gpu  ] $(nvidia-smi -L 2>/dev/null | head)"

module load python/miniforge3_pytorch/2.11.0

echo "[setup] ensuring jax[cuda12]==0.10.0 in user site-packages"
pip install --user --quiet "jax[cuda12]==0.10.0"
export PYTHONPATH=/u/jlyu5/.local/lib/python3.12/site-packages:${PYTHONPATH:-}

python -c "import jax; print('[setup] jax', jax.__version__, 'devices:', jax.devices())"

cd /u/jlyu5/factor_diffusion/1d_physics/1d_burgers/data_generation

OUT_PATH=/work/hdd/bgxp/factor_diffusion/original_data/burgers_1d/burgers_1d_test.pt
N_SAMPLES=500
BATCH_SIZE=200
SEED=2026

echo "[run  ] generate_dataset.py --n_samples ${N_SAMPLES} --batch_size ${BATCH_SIZE} --seed ${SEED} --out_path ${OUT_PATH}"
python generate_dataset.py \
    --n_samples "${N_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --out_path "${OUT_PATH}"

echo "[done ] $(date)"
