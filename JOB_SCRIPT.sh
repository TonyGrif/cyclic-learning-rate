#!/bin/bash
#SBATCH --job-name=clr
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -C v100
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -e

enable_lmod

module load container_env pytorch-gpu/2.7.1

srun crun -p ~/envs/clr python3 train.py "$1"
