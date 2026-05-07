#!/bin/bash
#SBATCH --job-name=CelebA_sample_eval
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/CelebA_HQ/k4_patch8_fixlr_Epoch878/logs/eval_batch256_%j.out
#SBATCH --error=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/CelebA_HQ/k4_patch8_fixlr_Epoch878/logs/eval_batch256_%j.err

set -euo pipefail

module --force purge
module load anaconda
source /anvil/scratch/x-<user>/envs/celeba_cu124/bin/activate

CKPT=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/CelebA_HQ/k4_patch8_fixlr_Epoch878/checkpoints/latest.pt
SAMPLE_DIR=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/CelebA_HQ/k4_patch8_fixlr_Epoch878/generated_10K_samples
EVAL_JSON=/anvil/scratch/x-<user>/physics_datasets/Experiments_Output/CelebA_HQ/k4_patch8_fixlr_Epoch878/eval/metrics_combined.json
SCRIPT_DIR=/anvil/scratch/x-<user>/physics_datasets/Experiments/CelebA_HQ

mkdir -p ${SAMPLE_DIR}
mkdir -p $(dirname ${EVAL_JSON})
mkdir -p /anvil/scratch/x-<user>/physics_datasets/Experiments_Output/CelebA_HQ/k4_patch8_fixlr_Epoch878/logs

echo "==============================="
echo "Step 1: Sampling (DDIM 250 steps, 10K images)"
echo "==============================="
python ${SCRIPT_DIR}/CelebA_DDIM_Sampling.py \
    --checkpoint        ${CKPT} \
    --output_dir        ${SAMPLE_DIR} \
    --pool_k            4 \
    --orig_size         1024 \
    --patch_size        8 \
    --hidden_dim        768 \
    --num_layers        12 \
    --num_heads         12 \
    --diff_timesteps    1000 \
    --num_samples       10000 \
    --batch_size        256 \
    --ddim_steps        250 \
    --eta               0.0

echo "==============================="
echo "Step 2: Evaluation (FID_InceptionV3 + FID_DINOv2 + Precision/Recall)"
echo "==============================="
python ${SCRIPT_DIR}/CelebA_Compute_metrics_combined.py \
    --gen-dir           ${SAMPLE_DIR} \
    --orig-dir          /anvil/scratch/x-<user>/physics_datasets/data/ground_truth_10K \
    --n-samples         10000 \
    --batch-size        128 \
    --knn-k             3 \
    --dino-model        dinov2_vitl14 \
    --dino-img-size     224 \
    --label             epoch_878 \
    --out-json          ${EVAL_JSON}

echo "==============================="
echo "All done! Results at: ${EVAL_JSON}"
echo "==============================="