#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=Reaction_k8_eval
#SBATCH --output=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/Reaction_1d/k8_fixlr_anvil/logs/eval_%j.out
#SBATCH --error=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/Reaction_1d/k8_fixlr_anvil/logs/eval_%j.err

module --force purge
module load anaconda
source /anvil/scratch/x-<user>/envs/celeba_cu124/bin/activate

echo "========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "========================================="

# ---- paths -------------------------------------------------------- #
CKPT=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/Reaction_1d/k8_fixlr_anvil/checkpoints/latest.pt
EVAL_SCRIPT=/anvil/scratch/x-<user>/physics_datasets/Experiments/Reaction_1d/Reaction_eval.py
OUTPUT_DIR=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/Reaction_1d/k8_fixlr_anvil

mkdir -p ${OUTPUT_DIR}

# ---- run ---------------------------------------------------------- #
python "$EVAL_SCRIPT" \
    --ckpt_path        "$CKPT"       \
    --test_path        /anvil/scratch/x-<user>/physics_datasets/data/reaction_1d/reaction_1d_test.pt \
    --output_dir       "$OUTPUT_DIR" \
    --pool_k           8             \
    --orig_L           1024          \
    --num_test_samples 500           \
    --num_seeds        5             \
    --hidden_dim       512           \
    --num_layers       12            \
    --num_heads        8             \
    --diff_timesteps   1000          \
    --ddim_steps       250

echo "========================================="
echo "Done : $(date)"
echo "========================================="