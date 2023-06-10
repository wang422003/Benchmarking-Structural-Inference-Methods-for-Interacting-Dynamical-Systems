#!/bin/bash -l
#SBATCH -J bsimds_ARACNE_hyper
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p big
#SBATCH --no-requeue

cd ~/bsimds/src/bio_models/ARACNE/hyper_parameter_tunning
conda activate ARACNE

srun -N 1 -n 1 --cpus-per-task 1 --unbuffered Rscript test.R

exit	
