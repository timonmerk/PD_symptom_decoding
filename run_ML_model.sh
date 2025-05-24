#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=medium
#SBATCH --time=7-00:00:00
#SBATCH -o logs/models.out
#SBATCH -e logs/models.err
#SBATCH -a 0-17

uv run run_decoding_ucsf_different_ML_methods.py $SLURM_ARRAY_TASK_ID