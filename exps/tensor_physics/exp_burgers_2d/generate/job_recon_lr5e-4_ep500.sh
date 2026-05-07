#!/bin/bash
#SBATCH -J burgers_lr5e-4_recon
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o /anvil/projects/x-eng260004/factor_diffusion/our_method_generation/burgers_2d_lr_sweep/lr_5e-4/logs/recon_%j.out
#SBATCH -e /anvil/projects/x-eng260004/factor_diffusion/our_method_generation/burgers_2d_lr_sweep/lr_5e-4/logs/recon_%j.err

OUTDIR=/anvil/projects/x-eng260004/factor_diffusion/our_method_generation/burgers_2d_lr_sweep/lr_5e-4

module load anaconda/2024.02-py311
source activate video_factor

cd /home/x-jlyu5/jinhua/factor_diffusion/tensor_physics

for s in 1 2 3 4; do
    python -u reconstruct_gen.py --exp burgers --seed $s --epoch 500 --dir $OUTDIR
done
