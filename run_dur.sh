#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=medium
#SBATCH --time=23:00:00
#SBATCH --constraint=skylake
#SBATCH -o logs/dur.out
#SBATCH -e logs/dur.err
#SBATCH -a 0

uv run how_much_data_is_needed.py $SLURM_ARRAY_TASK_ID