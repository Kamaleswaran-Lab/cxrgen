#!/bin/bash
#SBATCH --job-name=EHRxCXR
#SBATCH --output=../out/EHRxCXR.log
#SBATCH --error=../out/EHRxCXR.err
#SBATCH --time=12:00:00
#SBATCH --partition=compalloc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
##SBATCH --gres=gpu:1

source /hpc/dctrl/ma618/torch/bin/activate
which python
python ~/cxrgen/src/data/match_ehr_images.py
