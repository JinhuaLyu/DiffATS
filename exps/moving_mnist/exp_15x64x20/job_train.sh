#!/bin/bash
#SBATCH -J tucker_15x64x20_train
#SBATCH -A <ACCOUNT>
#SBATCH -p gengpu
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH -t 48:00:00
#SBATCH -o ${OUTPUT_ROOT}/moving_mnist/exp_15x64x20_output/logs/train_%j.out
#SBATCH -e ${OUTPUT_ROOT}/moving_mnist/exp_15x64x20_output/logs/train_%j.err

set -euo pipefail

mkdir -p ${OUTPUT_ROOT}/moving_mnist/exp_15x64x20_output/logs

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

python -u train.py train.yaml
