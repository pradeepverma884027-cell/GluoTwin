#!/bin/bash
#SBATCH --job-name=glu_lgbm
#SBATCH --partition=YOUR_CPU_PARTITION      # from: sinfo
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
module load python                            # name differs per cluster (module avail)
source ~/envs/glu/bin/activate
cd $SLURM_SUBMIT_DIR
python 01_baselines.py
python 02_train_lgbm.py --n_jobs $SLURM_CPUS_PER_TASK
