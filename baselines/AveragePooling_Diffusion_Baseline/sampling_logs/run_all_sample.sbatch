#!/bin/bash
#SBATCH --account=eng260004-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --job-name=avgpool_sample_4pdes
#SBATCH --output=/home/x-jlyu5/jinhua/Factor_Diffusion_Experiments/AveragePooling_Diffusion_Baseline/sampling_logs/sample_%j.out
#SBATCH --error=/home/x-jlyu5/jinhua/Factor_Diffusion_Experiments/AveragePooling_Diffusion_Baseline/sampling_logs/sample_%j.err

set -e

PYTHON=/anvil/projects/x-eng260004/factor_diffusion/diffusion_env/bin/python
ROOT=/home/x-jlyu5/jinhua/Factor_Diffusion_Experiments/AveragePooling_Diffusion_Baseline

echo "========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "========================================="
nvidia-smi || true

echo ""
echo "===== [1/4] Burgers_1D (k=4) ====="
$PYTHON "$ROOT/Burgers_1D/k4_lr1e-4/files/Burgers1d_sample.py"

echo ""
echo "===== [2/4] Reaction_1D (k=8) ====="
$PYTHON "$ROOT/Reaction_1D/k8_lr1e-4/files/Reaction1d_sample.py"

echo ""
echo "===== [3/4] Burgers_2D (k=8) ====="
$PYTHON "$ROOT/Burgers_2D/k8_lr1e-4/files/Burgers2d_sample.py"

echo ""
echo "===== [4/4] Karman_Vortex_2D (k=8) ====="
$PYTHON "$ROOT/Karman_Vortex_2D/k8_lr1e-4/files/Karman2d_sample.py"

echo ""
echo "========================================="
echo "All done : $(date)"
echo "========================================="
