#!/bin/bash
#SBATCH --job-name=data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=10000M
#SBATCH --partition=gpu_shared

#SRC_DIR=monet
#DATA_DIR=data
#DATASET=circles
#SAMPLES=50000

SRC_DIR=monet
DATA_DIR=data
DATASET=sprites_multi
SAMPLES=200000

# Copy data to scratch
cp -r $HOME/$SRC_DIR/$DATA_DIR $TMPDIR
cd $TMPDIR/$DATA_DIR

# Run experiment
source activate pytorch
srun python -u data.py generate_$DATASET with num_samples=$SAMPLES

cp -r $TMPDIR/$DATA_DIR/$DATASET $HOME/$SRC_DIR/$DATA_DIR
