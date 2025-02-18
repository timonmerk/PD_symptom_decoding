#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -o logs/norm_wo_hour.out
#SBATCH -e logs/norm_wo_hour.err
#SBATCH -a 0-41

uv run run_decoding_ucsf_across_patients_diff_norm_windows.py $SLURM_ARRAY_TASK_ID 0