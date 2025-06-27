#!/usr/bin/env bash
#SBATCH -J equilibrate
#SBATCH -p standard
#SBATCH -t 7-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --account=dmobley_lab
#SBATCH --output slurm-%x.%A.out

. ~/.bashrc

# Use the right conda environment
conda activate evaluator-050


python equilibrate-slurm.py                                             \
    --port                      8108                                    \
    --n-molecules               1000                                    \
    --extra-script-option       "--gpus-per-task=1"                     \
    --queue                     "free-gpu"                              \
    --n-gpu                     23                                      \
    --dataset                   "../training-properties-with-water.json"
    
