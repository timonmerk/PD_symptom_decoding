#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=medium
#SBATCH --time=7-00:00:00
#SBATCH -o logs/nighttime.out
#SBATCH -e logs/nighttime.err
#SBATCH -a 11

uv run run_decoding_ucsf_across_patients_nighttime.py $SLURM_ARRAY_TASK_ID