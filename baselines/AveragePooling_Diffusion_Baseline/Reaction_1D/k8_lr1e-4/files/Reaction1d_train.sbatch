#!/bin/bash
#SBATCH --job-name=Reaction1d_k8_fixlr
#SBATCH --account=eng260004-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/Reaction_1d/k8_fixlr_anvil/logs/0501_train_%j.out
#SBATCH --error=/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/Reaction_1d/k8_fixlr_anvil/logs/0501_train_%j.err

module --force purge
module load anaconda
source /anvil/scratch/x-ezhou1/envs/celeba_cu124/bin/activate

OUTPUT_DIR=/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/Reaction_1d/k8_fixlr_anvil

mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${OUTPUT_DIR}/checkpoints

cd /anvil/scratch/x-ezhou1/physics_datasets/Experiments/Reaction_1d

export WANDB_ENTITY=eleanorz924-northwestern-university
export WANDB_PROJECT=reaction1d_k8_fixlr_anvil_0501

python /anvil/scratch/x-ezhou1/physics_datasets/Experiments/Reaction_1d/Reaction_train.py \
    --data_path         /anvil/scratch/x-ezhou1/physics_datasets/data/reaction_1d/reaction_1d_train.pt \
    --pool_k            8 \
    --hidden_dim        512 \
    --num_layers        12 \
    --num_heads         8 \
    --diff_timesteps    1000 \
    --epochs            1000 \
    --batch_size        32 \
    --lr                1e-4 \
    --num_workers       8 \
    --grad_clip         1.0 \
    --output_dir        ${OUTPUT_DIR} \
    --save_every        50 \
    --log_every_steps   100 \
    --sample_every_epochs 10 \
    --num_sample_plots  3 \
    --wandb_project     "reaction1d_k8_fixlr_anvil_0501" \
    --resume            ${OUTPUT_DIR}/checkpoints/latest.pt
