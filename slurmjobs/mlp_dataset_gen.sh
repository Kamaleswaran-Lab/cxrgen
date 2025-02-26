#!/bin/bash
#SBATCH --job-name=mlp_dataset
#SBATCH --output=../out/mlp_dataset.log
#SBATCH --error=../out/mlp_dataset.err
#SBATCH --time=24:00:00
#SBATCH --partition=compalloc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --nodes=1
##SBATCH --gres=gpu:1

source /hpc/dctrl/ma618/torch/bin/activate
which python
python ~/cxrgen/src/data/mlp_dataset_rowwise.py
