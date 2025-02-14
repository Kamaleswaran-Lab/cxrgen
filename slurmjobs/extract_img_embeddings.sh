#!/bin/bash
#SBATCH --job-name=img_embeddings
#SBATCH --output=../out/extract_img_embeddings.log
#SBATCH --error=../out/extract_img_embeddings.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-common
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

source /hpc/dctrl/ma618/torch/bin/activate
which python
python ~/cxrgen/src/data/image_embeddings.py