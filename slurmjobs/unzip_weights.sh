#!/bin/bash
#SBATCH --job-name=weights
#SBATCH --output=../out/extract_weights.log
#SBATCH --error=../out/extract_weights.err
#SBATCH --time=24:00:00
#SBATCH --partition=compalloc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

path_to_destination="/hpc/group/kamaleswaranlab/cxrgen/roentgen"
weights_file='/hpc/home/ma618/cxrgen/roentgen_v1.0.zip'

echo "Extracting $weights_file to $path_to_destination"

unzip -qq $weights_file -d $path_to_destination

echo "Extraction complete."