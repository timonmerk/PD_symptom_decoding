#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=medium
#SBATCH --time=23:00:00
#SBATCH -o logs/models.out
#SBATCH -e logs/models.err
#SBATCH -a 0-6

uv run run_decoding_ucsf_different_ML_methods.py $SLURM_ARRAY_TASK_ID