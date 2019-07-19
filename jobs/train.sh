#!/bin/bash
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=20000M
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

# circles: ~ 68 steps/second

SRC_DIR=monet

# Copy data to scratch
cp -r $HOME/$SRC_DIR $TMPDIR
cd $TMPDIR/$SRC_DIR

# Run experiment
source activate pytorch
srun python -u train.py with dataset='circles' lr=0.0001 steps=100000
