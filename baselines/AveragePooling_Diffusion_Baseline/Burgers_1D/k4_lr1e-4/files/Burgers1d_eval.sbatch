#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=Burgers1d_k4_eval
#SBATCH --output=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/Burgers_1d/k4_fixlr/logs/eval_%j.out
#SBATCH --error=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/Burgers_1d/k4_fixlr/logs/eval_%j.err

module --force purge
module load anaconda
source /anvil/scratch/x-<user>/envs/celeba_cu124/bin/activate

echo "========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "========================================="


CKPT=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/Burgers_1d/k4_fixlr/checkpoints/latest.pt
EVAL_SCRIPT=/anvil/scratch/x-<user>/physics_datasets/Experiments/Burgers_1d/Burgers_eval.py
TEST_DATA=/anvil/scratch/x-<user>/physics_datasets/data/burgers_1d/burgers_1d_test.pt
OUTPUT_DIR=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/Burgers_1d/k4_fixlr

mkdir -p "$OUTPUT_DIR"

# ---- run ---------------------------------------------------------- #
python "$EVAL_SCRIPT" \
    --ckpt_path      "$CKPT"       \
    --test_path      "$TEST_DATA"  \
    --output_dir     "$OUTPUT_DIR" \
    --pool_k         4             \
    --orig_L         1024          \
    --num_test_samples 500         \
    --num_seeds      5             \
    --hidden_dim     512           \
    --num_layers     12            \
    --num_heads      8             \
    --diff_timesteps 1000          \
    --ddim_steps     250

echo "========================================="
echo "Done : $(date)"
echo "========================================="