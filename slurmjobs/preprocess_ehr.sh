#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --output=../out/ehr_preprocess2.log
#SBATCH --error=../out/ehr_preprocess2.err
#SBATCH --time=24:00:00
#SBATCH --partition=compalloc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --nodes=1

source /hpc/dctrl/ma618/torch/bin/activate

python ~/cxrgen/src/data/ehr_preprocessing.py