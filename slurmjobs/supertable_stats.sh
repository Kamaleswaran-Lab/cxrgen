#!/bin/bash
#SBATCH --job-name=supertable_stats
#SBATCH --output=../out/supertable_stats.log
#SBATCH --error=../out/supertable_stats.err
#SBATCH --time=24:00:00
#SBATCH --partition=compalloc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --nodes=1

source /hpc/dctrl/ma618/torch/bin/activate

python ~/cxrgen/src/data/get_supertable_variable_stats.py