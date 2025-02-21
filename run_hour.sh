#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=medium
#SBATCH --time=20:00:00
#SBATCH --constraint=haswell|broadwell|skylake
#SBATCH -o logs/hour.out
#SBATCH -e logs/hour.err
#SBATCH -a 0

uv run run_decoding_ucsf_across_patients_hour.py $SLURM_ARRAY_TASK_ID