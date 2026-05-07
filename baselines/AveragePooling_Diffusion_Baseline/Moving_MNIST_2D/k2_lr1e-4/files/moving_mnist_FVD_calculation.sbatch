#!/bin/bash
#SBATCH --account=eng260004-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --job-name=Mnist_gen_fvd
#SBATCH --output=/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/Moving_Mnist/k2_fixlr_A100/logs/gen_fvd_%j.out
#SBATCH --error=/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/Moving_Mnist/k2_fixlr_A100/logs/gen_fvd_%j.err

module --force purge
module load anaconda
source /anvil/scratch/x-ezhou1/envs/celeba_cu124/bin/activate

# dirs
SCRIPT_DIR=/anvil/scratch/x-ezhou1/physics_datasets/Experiments/Moving_Mnist
CKPT=/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/Moving_Mnist/k2_fixlr_A100/checkpoints/latest.pt
DATA=/anvil/scratch/x-ezhou1/physics_datasets/data/moving_mnist/moving_mnist_20k_2slow.pt
OUT_DIR=/anvil/scratch/x-ezhou1/physics_datasets/Experiments_Output/Moving_Mnist/k2_fixlr_A100
GEN_OUT=${OUT_DIR}/generated_10k.pt
FVD_OUT=${OUT_DIR}/fvd_results/fvd_report.json
I3D=/anvil/scratch/x-ezhou1/physics_datasets/Experiments/Moving_Mnist/i3d_torchscript.pt  

mkdir -p ${OUT_DIR}/logs
mkdir -p ${OUT_DIR}/fvd_results

echo "========================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Started at  : $(date)"
echo "========================================="

# ── Step 1: Generate 10K videos ───────────────────────────────────────────────
echo ""
echo "[Step 1/2] Generating 10K videos with DDIM-250..."
python ${SCRIPT_DIR}/MovingMnist_generation.py \
    --ckpt        ${CKPT} \
    --out         ${GEN_OUT} \
    --n_generate  10000 \
    --batch_size  64 \
    --ddim_steps  250 \
    --eta         0.0

if [ $? -ne 0 ]; then
    echo "ERROR: Generation failed. Exiting."
    exit 1
fi

echo "Generation complete: ${GEN_OUT}"

# ── Step 2: Compute FVD ───────────────────────────────────────────────────────
echo ""
echo "[Step 2/2] Computing FVD..."
python ${SCRIPT_DIR}/MovingMnist_FVD_Calculation.py \
    --real        ${DATA} \
    --gen         ${GEN_OUT} \
    --n           10000 \
    --batch       32 \
    --i3d         ${I3D} \
    --real_select random \
    --seed        42 \
    --out         ${FVD_OUT}

if [ $? -ne 0 ]; then
    echo "ERROR: FVD computation failed. Exiting."
    exit 1
fi

echo ""
echo "========================================="
echo "FVD report: ${FVD_OUT}"
echo "Finished at: $(date)"
echo "========================================="
