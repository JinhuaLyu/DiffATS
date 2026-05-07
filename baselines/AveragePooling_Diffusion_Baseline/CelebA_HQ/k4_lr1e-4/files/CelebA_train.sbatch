#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=CelebA_k4patch8__Epoch870
#SBATCH --output=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/CelebA_HQ/k4_patch8_fixlr_Epoch878/logs/Epoch878_train_%j.out
#SBATCH --error=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/CelebA_HQ/k4_patch8_fixlr_Epoch878/logs/Epoch878_train_%j.err

module --force purge
module load anaconda
source /anvil/scratch/x-<user>/envs/celeba_cu124/bin/activate

OUTPUT_DIR=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/CelebA_HQ/k4_patch8_fixlr_Epoch878

mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${OUTPUT_DIR}/checkpoints

cd /anvil/scratch/x-<user>/physics_datasets/Experiments/CelebA_HQ

export WANDB_ENTITY=<WANDB_ENTITY>
export WANDB_PROJECT=celeba_hq_k4_patch8_fixlr_anvil_Epoch878

python /anvil/scratch/x-<user>/physics_datasets/Experiments/CelebA_HQ/CelebA_train.py \
    --data_dir          /anvil/scratch/x-<user>/physics_datasets/data/all \
    --pool_k            4 \
    --orig_size         1024 \
    --patch_size        8 \
    --hidden_dim        768 \
    --num_layers        12 \
    --num_heads         12 \
    --diff_timesteps    1000 \
    --epochs            1000 \
    --batch_size        32 \
    --lr                1e-4 \
    --num_workers       16 \
    --grad_clip         1.0 \
    --output_dir        ${OUTPUT_DIR} \
    --save_every        50 \
    --log_every_steps   100 \
    --sample_every_epochs 50 \
    --num_sample_plots  3 \
    --wandb_project     "celeba_hq_k4_patch8_fixlr_anvil_Epoch878" \
    --resume            /anvil/scratch/x-<user>/physics_datasets/Experiments_Output/CelebA_HQ/k4_patch8_fixlr/checkpoints/latest_Epoch870.pt
