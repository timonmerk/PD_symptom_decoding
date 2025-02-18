#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -o logs/hour.out
#SBATCH -e logs/hour.err
#SBATCH -a 0-11

uv run run_decoding_ucsf_across_patients_hour.py $SLURM_ARRAY_TASK_ID