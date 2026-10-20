#!/bin/bash
# usage:  sbatch job_gpu.sh        (trains the GRU on both time-series datasets)
#SBATCH --job-name=glu_gru
#SBATCH --partition=YOUR_GPU_PARTITION      # see: sinfo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
module load python cuda                       # names differ per cluster
source ~/envs/glu/bin/activate
cd $SLURM_SUBMIT_DIR
python train_gru.py --raw_dir data --epochs 60
