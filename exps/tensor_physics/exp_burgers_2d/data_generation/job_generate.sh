#!/bin/bash
#SBATCH -A p32954
#SBATCH -J burgers2d_gen
#SBATCH -p gengpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /projects/p32954/jinhua_data/burgers_2d/logs/gen_%j.out
#SBATCH -e /projects/p32954/jinhua_data/burgers_2d/logs/gen_%j.err

mkdir -p /projects/p32954/jinhua_data/burgers_2d/logs

cd /gpfs/home/fzd2816/factor_diffusion/tensor_physics/exp_burgers_2d/data_generation

/home/fzd2816/.conda/envs/video_factor/bin/python generate_data.py
