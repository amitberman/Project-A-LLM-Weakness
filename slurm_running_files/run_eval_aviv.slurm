#!/bin/bash

### Job Configuration ###

# Job name
#SBATCH --job-name=human_eval_fix

# Output/Error files
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.out

# Partition: CHANGED to 'student-batch' 
# (The 'debug' partition has a 15-minute limit, so 6 hours would be rejected or killed)
#SBATCH --partition=student-batch

# Request 1 GPU
#SBATCH --gpus=1

# Resources
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=6:00:00

### Execution Section ###

echo "Job started on $(hostname) at $(date)"

# --- FIX 1: Correct Conda Activation ---
# Do not source .bashrc or 'activate' a .sh file. 
# Source the conda script, THEN activate the environment.
source /home/projects/hsdsl-prj10212/anaconda3/etc/profile.d/conda.sh
conda activate base

# --- FIX 2: Handle the "Core Dumped" Crash ---
# The crash happens because GPT4All is fighting with the cluster's GPU drivers.
# We try to force it to use the CUDA backend explicitly.
export GPT4ALL_BACKEND="cuda"

# Debug print to confirm python version
which python

# 3. Run the Python script
python -u /home/projects/hsdsl-prj10212/Project-A-LLM-Weakness/he_test3.py

echo "Job finished at $(date)"