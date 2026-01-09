#!/bin/bash

### Job Configuration ###

# Job name (will show up in squeue)
#SBATCH --job-name=human_eval_gpu

# Output file for standard print outputs (%j will be replaced by job ID)
#SBATCH --output=slurm_logs/slurm-%j.out

# Error file for error logs (useful for debugging)
#SBATCH --error=slurm_logs/slurm-%j.out

# Partition/Queue name (Usually 'student-batch' or 'gpu' in Technion clusters)
#SBATCH --partition=debug

# Request 1 GPU
#SBATCH --gpus=1

# Request 2 CPU cores (sufficient for data loading)
#SBATCH --cpus-per-task=2

# Request 16GB of RAM (System memory, not GPU memory)
#SBATCH --mem=16G

# Time limit: 3 hours (Safe buffer for 160 questions)
#SBATCH --time=3:00:00


### Execution Section ###

echo "Job started on $(hostname) at $(date)"

# Load user bash settings to enable conda
source ~/.bashrc

#Activate your conda environment
conda activate /home/projects/hsdsl-prj10212/anaconda3/etc/profile.d/conda.sh

# Ensure correct path to conda.sh
conda init
conda activate base

# 3. Run the Python script
python -u /home/projects/hsdsl-prj10212/Project-A-LLM-Weakness/assitant_he_tests.py

echo "Job finished at $(date)"