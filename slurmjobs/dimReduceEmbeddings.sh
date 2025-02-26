#!/bin/bash
#SBATCH --job-name=SdimReduceEmbeddings
#SBATCH --output=../out/dim_reduce_embeddings_selected.log
#SBATCH --error=../out/dim_reduce_embeddings_selected.err
#SBATCH --time=24:00:00
#SBATCH --partition=scavenger-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

source /hpc/dctrl/ma618/rapids/bin/activate
which python
python ~/cxrgen/src/dimensionalityReduction.py