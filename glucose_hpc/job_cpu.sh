#!/bin/bash
# usage:  sbatch --export=ALL,MODEL=lgbm job_cpu.sh      (MODEL = linear | rf | lgbm | xgb)
#SBATCH --job-name=glu_cpu
#SBATCH --partition=YOUR_CPU_PARTITION      # see: sinfo
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
module load python                            # name differs per cluster: module avail
source ~/envs/glu/bin/activate
cd $SLURM_SUBMIT_DIR
python train_${MODEL}.py --raw_dir data --n_jobs $SLURM_CPUS_PER_TASK
