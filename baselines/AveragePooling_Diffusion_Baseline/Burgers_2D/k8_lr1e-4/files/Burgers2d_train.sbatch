#!/bin/bash
#SBATCH --job-name=Burgers_Avgpool_k8_H200_fixlr
#SBATCH --account=<group>-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=10:00:00
#SBATCH --output=${DATA_ROOT}/Experiments_Output/Burgers_2D/k8_H200_fixlr/logs/fixlr_train_%j.out
#SBATCH --error=${DATA_ROOT}/Experiments_Output/Burgers_2D/k8_H200_fixlr/logs/fixlr_train_%j.err

set -euo pipefail

export PATH="/u/<user>/.conda/envs/CelebA_cu124/bin:$PATH"

OUTPUT_DIR=${DATA_ROOT}/Experiments_Output/Burgers_2D/k8_H200_fixlr

mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${OUTPUT_DIR}/checkpoints

cd ${DATA_ROOT}/Experiments/Burgers_2d

export WANDB_ENTITY=<WANDB_ENTITY>
export WANDB_PROJECT=Burgers2d_avgpool_k8_lr1e-4_H200_fixlr

python ${DATA_ROOT}/Experiments/Burgers_2d/Burgers2d_train.py \
    --shard_dir             ${DATA_ROOT}/data/burgers_2d \
    --pool_k                8 \
    --total_timesteps_data  201 \
    --samples_per_shard     100 \
    --hidden_dim            512 \
    --num_layers            12 \
    --fields                ux uy \
    --num_heads             8 \
    --diff_timesteps        1000 \
    --epochs                500 \
    --batch_size            32 \
    --lr                    1e-4 \
    --num_workers           16 \
    --grad_clip             1.0 \
    --output_dir            ${OUTPUT_DIR} \
    --save_every            50 \
    --log_every_steps       100 \
    --sample_every_epochs   10 \
    --num_sample_plots      3 \
    --wandb_project         "Burgers2d_avgpool_k8_lr1e-4_H200_fixlr" \
    --resume                ${OUTPUT_DIR}/checkpoints/latest.pt
