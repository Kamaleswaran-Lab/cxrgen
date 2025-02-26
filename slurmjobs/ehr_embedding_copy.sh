#!/bin/bash
#SBATCH --job-name=EHRCXR_copy
#SBATCH --output=../out/EHRCXR_copy.log
#SBATCH --error=../out/EHRCXR_copy.err
#SBATCH --time=12:00:00
#SBATCH --partition=compalloc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --nodes=1
##SBATCH --gres=gpu:1

source /hpc/dctrl/ma618/torch/bin/activate
which python
python ~/cxrgen/src/data/move_longitudinal_data.py
