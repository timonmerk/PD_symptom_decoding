#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -o logs/loso.out
#SBATCH -e logs/loso.err
#SBATCH -a 0-5

uv run leave_one_subject_out_not_hemisphere_validation.py $SLURM_ARRAY_TASK_ID