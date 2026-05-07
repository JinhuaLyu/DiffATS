#!/bin/bash
#SBATCH -J burgers_lr5e-4_gen
#SBATCH -A <ACCOUNT>
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 1:30:00
#SBATCH -o ${DATA_ROOT}/our_method_generation/burgers_2d_lr_sweep/lr_5e-4/logs/gen_%j.out
#SBATCH -e ${DATA_ROOT}/our_method_generation/burgers_2d_lr_sweep/lr_5e-4/logs/gen_%j.err

OUTDIR=${DATA_ROOT}/our_method_generation/burgers_2d_lr_sweep/lr_5e-4
mkdir -p $OUTDIR/logs

module load anaconda/2024.02-py311
source activate video_factor

cd ${REPO_ROOT}/tensor_physics/exp_burgers_2d/generate

python -u gen_burgers_2d.py \
    --ckpt ${DATA_ROOT}/our_method_results/burgers_2d_lr_sweep/lr_5e-4/checkpoints/epoch00500_step0156000.pt \
    --output_dir $OUTDIR \
    --batch_size 50 \
    --seeds 0 1 2 3 4 \
    --sample_steps 250 \
    --device cuda:0 \
    --epoch 500

cd ${REPO_ROOT}/tensor_physics
python -u reconstruct_gen.py --exp burgers --seed 0 --epoch 500 --dir $OUTDIR
