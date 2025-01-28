#!/bin/bash
#SBATCH --job-name=extract_tar
#SBATCH --output=../out/extract_tar_%a.log
#SBATCH --error=../out/extract_tar_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=compalloc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --array=0-6

YEARS=(2016 2017 2018 2019 2020 2021 2022)

TAR_DIR="/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays/"
EXTRACT_DIR="/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays/"

# Spawn jobs for each year
year=${YEARS[$SLURM_ARRAY_TASK_ID]}
tar_file="${TAR_DIR}${year}_archive.tar.gz"
EXTRACT_FOLDER="${EXTRACT_DIR}${year}/"

mkdir -p $EXTRACT_FOLDER 

echo "Extracting $tar_file to $EXTRACT_DIR"

tar -xvf $tar_file -C $EXTRACT_DIR

echo "Extraction complete."