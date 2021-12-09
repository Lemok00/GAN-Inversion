#!/bin/bash
#SBATCH -o ../logs/211020_FFHQ256.out
#SBATCH -J FFHQ256
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --nodelist=node07


nvidia-smi

cd ../train

python train_restyle_encoder_0805.py --name AfhqCat256
