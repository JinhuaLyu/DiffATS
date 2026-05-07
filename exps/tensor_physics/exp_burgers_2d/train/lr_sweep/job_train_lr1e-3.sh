#!/bin/bash
#SBATCH -J burgers_lr1e-3
#SBATCH -A <ACCOUNT>
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 05:00:00
#SBATCH -o ${DATA_ROOT}/our_method_results/burgers_2d_lr_sweep/logs/lr1e-3_%j.out
#SBATCH -e ${DATA_ROOT}/our_method_results/burgers_2d_lr_sweep/logs/lr1e-3_%j.err

mkdir -p ${DATA_ROOT}/our_method_results/burgers_2d_lr_sweep/logs

module load anaconda/2024.02-py311
source activate video_factor

cd ${REPO_ROOT}/tensor_physics/exp_burgers_2d/train

python -u train_burgers_2d.py \
    --config ${REPO_ROOT}/tensor_physics/exp_burgers_2d/configs/train_v1.yaml \
    --lr 1e-3 \
    --n_epochs 500 \
    --outdir ${DATA_ROOT}/our_method_results/burgers_2d_lr_sweep/lr_1e-3 \
    --wandb_run lrsweep_burgers2d_lr1e-3
