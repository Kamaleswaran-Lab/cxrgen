#!/bin/bash
#SBATCH --job-name=EmbA
#SBATCH --output=../out/analyzeEmbeddings.log
#SBATCH --error=../out/analyzeEmbeddings.err
#SBATCH --time=24:00:00
#SBATCH --partition=compalloc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --nodes=1
##SBATCH --gres=gpu:1

source /hpc/dctrl/ma618/torch/bin/activate
which python
python ~/cxrgen/src/data/analyze_embeddings.py
