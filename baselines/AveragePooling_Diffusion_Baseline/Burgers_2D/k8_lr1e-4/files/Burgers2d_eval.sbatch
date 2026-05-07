#!/bin/bash
#SBATCH --account=eng260004-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --job-name=Burgers2d_eval
#SBATCH --output=/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/Burgers_2D/k8_H200_fixlr/logs/eval_%j.out
#SBATCH --error=/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/Burgers_2D/k8_H200_fixlr/logs/eval_%j.err

module --force purge
module load anaconda
source /anvil/scratch/x-ezhou1/envs/celeba_cu124/bin/activate

OUTPUT_DIR=/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/Burgers_2D/k8_H200_fixlr

mkdir -p ${OUTPUT_DIR}/logs

cd /anvil/scratch/x-ezhou1/physics_datasets/Experiments/Burgers_2d

python /anvil/scratch/x-ezhou1/physics_datasets/Experiments/Burgers_2d/Burgers2d_eval.py \
    --test_dir          /anvil/scratch/x-ezhou1/physics_datasets/data/burgers_2d/test_data \
    --pool_k            8 \
    --orig_size         128 \
    --num_test_samples  500 \
    --samples_per_shard 100 \
    --num_seeds         5 \
    --ckpt_path         ${OUTPUT_DIR}/checkpoints/final.pt \
    --hidden_dim        512 \
    --num_layers        12 \
    --num_heads         8 \
    --diff_timesteps    1000 \
    --ddim_steps        250 \
    --output_dir        ${OUTPUT_DIR}
