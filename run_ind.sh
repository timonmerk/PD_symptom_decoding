#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -o logs/ind.out
#SBATCH -e logs/ind.err
#SBATCH -a 0-119

uv run figure_27_get_ch_ind_per_all_ch.py $SLURM_ARRAY_TASK_ID