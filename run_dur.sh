#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -o logs/dur.out
#SBATCH -e logs/dur.err
#SBATCH -a 0-38

uv run how_much_data_is_needed.py $SLURM_ARRAY_TASK_ID