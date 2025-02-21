#!/bin/bash
#SBATCH --job-name=dimReduceEmbeddings
#SBATCH --output=../out/dim_reduce_embeddings.log
#SBATCH --error=../out/dim_reduce_embeddings.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-common
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

source /hpc/dctrl/ma618/rapids/bin/activate
which python
python ~/cxrgen/src/dimensionalityReduction.py