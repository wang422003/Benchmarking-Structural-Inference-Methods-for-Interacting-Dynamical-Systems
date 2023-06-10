#!/bin/bash -l
#SBATCH -J bsimds_CLR_hyper
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p big
#SBATCH --no-requeue

cd ~/bsimds/src/bio_models/CLR/hyper_parameter_tunning
conda activate CLR

srun -N 1 -n 1 --cpus-per-task 1 --unbuffered Rscript test.R

exit	
