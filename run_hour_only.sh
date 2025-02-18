#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -o logs/hour_only.out
#SBATCH -e logs/hour_only.err
#SBATCH -a 0-5

uv run run_decoding_ucsf_across_patients_hour_only.py $SLURM_ARRAY_TASK_ID