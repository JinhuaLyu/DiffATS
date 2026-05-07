#!/bin/bash
#SBATCH --account=eng260004-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=avgpool_karman_sample
#SBATCH --output=/home/x-jlyu5/jinhua/Factor_Diffusion_Experiments/AveragePooling_Diffusion_Baseline/sampling_logs/karman_%j.out
#SBATCH --error=/home/x-jlyu5/jinhua/Factor_Diffusion_Experiments/AveragePooling_Diffusion_Baseline/sampling_logs/karman_%j.err

set -e

PYTHON=/anvil/projects/x-eng260004/factor_diffusion/diffusion_env/bin/python
ROOT=/home/x-jlyu5/jinhua/Factor_Diffusion_Experiments/AveragePooling_Diffusion_Baseline

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
nvidia-smi || true

echo "===== Karman_Vortex_2D (k=8) ====="
$PYTHON "$ROOT/Karman_Vortex_2D/k8_lr1e-4/files/Karman2d_sample.py"

echo "Done : $(date)"
