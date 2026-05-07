#!/bin/bash
#SBATCH --job-name=MNIST_k2_fixlr
#SBATCH --account=bgxp-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/bgxp/ezhou1/factor_diffusion_proj/Experiments_Output/Moving_Mnist/k2_fixlr/logs/H200_train_%j.out
#SBATCH --error=/scratch/bgxp/ezhou1/factor_diffusion_proj/Experiments_Output/Moving_Mnist/k2_fixlr/logs/H200_train_%j.err

set -euo pipefail

export PATH="/u/ezhou1/.conda/envs/CelebA_cu124/bin:$PATH"

OUTPUT_DIR=/scratch/bgxp/ezhou1/factor_diffusion_proj/Experiments_Output/Moving_Mnist/k2_fixlr

mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${OUTPUT_DIR}/checkpoints

cd /scratch/bgxp/ezhou1/factor_diffusion_proj/Experiments/Moving_Mnist

export WANDB_ENTITY=eleanorz924-northwestern-university
export WANDB_PROJECT=moving_mnist_k2_fixlr

python /scratch/bgxp/ezhou1/factor_diffusion_proj/Experiments/Moving_Mnist/MovingMnist_train_full_preload_H200_fixlr.py \
    --data_path         /scratch/bgxp/ezhou1/factor_diffusion_proj/data/moving_mnist/moving_mnist_20k_2slow.pt \
    --pool_k            2 \
    --hidden_dim        512 \
    --num_layers        12 \
    --num_heads         8 \
    --diff_timesteps    1000 \
    --epochs            2000 \
    --batch_size        32 \
    --lr                1e-4 \
    --num_workers       8 \
    --grad_clip         1.0 \
    --output_dir        ${OUTPUT_DIR} \
    --save_every        50 \
    --log_every_steps   100 \
    --sample_every_epochs 10 \
    --num_sample_plots  3 \
    --wandb_project     "moving_mnist_k2_fixlr" \
    --resume            ${OUTPUT_DIR}/checkpoints/latest.pt
