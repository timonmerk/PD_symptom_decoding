#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -o logs/nighttime.out
#SBATCH -e logs/nighttime.err
#SBATCH -a 0-11

uv run run_decoding_ucsf_across_patients_nighttime.py $SLURM_ARRAY_TASK_ID