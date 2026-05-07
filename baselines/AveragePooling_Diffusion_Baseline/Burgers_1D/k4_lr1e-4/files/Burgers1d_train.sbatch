#!/bin/bash
#SBATCH --account=eng260004-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=Burgers1d_k4_train
#SBATCH --output=/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/Burgers_1d/train_%j.out


module --force purge
module load anaconda
source /anvil/scratch/x-ezhou1/envs/celeba_cu124/bin/activate

OUTPUT_DIR=/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/Burgers_1d

mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${OUTPUT_DIR}/checkpoints

cd /anvil/scratch/x-ezhou1/physics_datasets/Experiments/Burgers_1d

export WANDB_ENTITY=eleanorz924-northwestern-university
export WANDB_PROJECT=burgers1d_k4_fixlr_anvil

python /anvil/scratch/x-ezhou1/physics_datasets/Experiments/Burgers_1d/Burgers_train.py \
    --data_path         /anvil/scratch/x-ezhou1/physics_datasets/data/burgers_1d/burgers_1d.pt \
    --pool_k            4 \
    --hidden_dim        512 \
    --num_layers        12 \
    --num_heads         8 \
    --diff_timesteps    1000 \
    --epochs            1000 \
    --batch_size        64 \
    --lr                1e-4 \
    --num_workers       8 \
    --grad_clip         1.0 \
    --output_dir        ${OUTPUT_DIR} \
    --save_every        50 \
    --log_every_steps   100 \
    --sample_every_epochs 10 \
    --num_sample_plots  3 \
    --wandb_project     "burgers1d_k4_fixlr_anvil" \
    --resume            ${OUTPUT_DIR}/checkpoints/latest.pt
