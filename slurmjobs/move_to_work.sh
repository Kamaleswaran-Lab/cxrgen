#!/bin/bash
#SBATCH --job-name=move_to_work
#SBATCH --output=../out/move_to_work.log
#SBATCH --error=../out/move_to_work.err
#SBATCH --time=24:00:00
#SBATCH --partition=compalloc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --nodes=1

source /hpc/dctrl/ma618/torch/bin/activate
which python
python ~/cxrgen/src/data/move_dataset_work.py