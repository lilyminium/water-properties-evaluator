#!/bin/bash
#SBATCH -J gather
#SBATCH -p standard
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --account dmobley_lab
#SBATCH --export ALL
#SBATCH --mem=16gb
#SBATCH --constraint=fastscratch
#SBATCH --output slurm-%x.%A.out


. ~/.bashrc

# Use the right conda environment
conda activate evaluator-050




python gather-benchmarks.py     \
    -i "../training-properties-with-water.json"                 \
    -d output                   \
    -o .
