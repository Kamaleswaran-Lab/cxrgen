#!/bin/bash
#SBATCH --job-name=YOLO
#SBATCH --output=../out/train_YOLO_mlp.log
#SBATCH --error=../out/train_YOLO_mlp.err
#SBATCH --time=24:00:00
#SBATCH --partition=scavenger-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

source /hpc/dctrl/ma618/torch/bin/activate
which python
python ~/cxrgen/src/train.py 