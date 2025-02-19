#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -o logs/main.out
#SBATCH -e logs/main.err
#SBATCH -a 0-17

uv run run_decoding_main_figure.py $SLURM_ARRAY_TASK_ID