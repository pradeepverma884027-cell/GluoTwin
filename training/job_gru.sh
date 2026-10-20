#!/bin/bash
#SBATCH --job-name=glu_gru
#SBATCH --partition=YOUR_GPU_PARTITION      # from: sinfo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
module load python cuda                       # names differ per cluster
source ~/envs/glu/bin/activate
cd $SLURM_SUBMIT_DIR
python 03_train_gru.py cgm_t1dm_t2dm --epochs 60
python 03_train_gru.py ohhim --epochs 100
