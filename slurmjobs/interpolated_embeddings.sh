#!/bin/bash
#SBATCH --job-name=interpolate
#SBATCH --output=../out/interpolate_image.log
#SBATCH --error=../out/interpolate_image.err
#SBATCH --time=24:00:00
#SBATCH --partition=compalloc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --nodes=1

source /hpc/dctrl/ma618/torch/bin/activate

python ~/cxrgen/src/data/image_interpolations.py