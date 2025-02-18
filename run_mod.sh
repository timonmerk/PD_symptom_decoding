#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -o logs/mod.out
#SBATCH -e logs/mod.err
#SBATCH -a 0-23

uv run run_decoding_ucsf_across_patients_diff_features.py $SLURM_ARRAY_TASK_ID