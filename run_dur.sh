#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=medium
#SBATCH --time=7-00:00:00
#SBATCH --constraint=haswell|broadwell|skylake
#SBATCH -o logs/dur.out
#SBATCH -e logs/dur.err
#SBATCH -a 0-38

uv run how_much_data_is_needed.py $SLURM_ARRAY_TASK_ID