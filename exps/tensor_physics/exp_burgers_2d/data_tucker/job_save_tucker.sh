#!/bin/bash
#SBATCH -A p32954
#SBATCH -J burgers2d_tucker
#SBATCH -p normal
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /projects/p32954/jinhua_output/burgers_2d/tucker_factors/logs/save_tucker_%j.out
#SBATCH -e /projects/p32954/jinhua_output/burgers_2d/tucker_factors/logs/save_tucker_%j.err

mkdir -p /projects/p32954/jinhua_output/burgers_2d/tucker_factors/logs

cd /home/fzd2816/factor_diffusion/tensor_physics/exp_burgers_2d/data_tucker

/home/fzd2816/.conda/envs/video_factor/bin/python save_tucker_burgers.py \
    --data_dir /projects/p32954/jinhua_data/burgers_2d \
    --out_dir  /projects/p32954/jinhua_output/burgers_2d/tucker_factors/tucker_burgers_rT5_rH20_rW20 \
    --rank_T 5 --rank_H 20 --rank_W 20 --r_ic 20 \
    --n_workers $SLURM_CPUS_PER_TASK
