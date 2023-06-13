#!/bin/bash -l
#SBATCH -J bsimds_scribe_hyper
#SBATCH --mail-type end,fail
#SBATCH --mail-user tszpan.tong@uni.lu
#SBATCH -N 64
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH --time 2-00:00:00
#SBATCH -p batch
#SBATCH --no-requeue

cd /home/users/ttong/scribe/hyper_parameter_tunning
conda activate scribe

srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 2 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 2 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 3 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 3 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 4 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 4 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 5 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 5 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 1 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 2 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 3 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 4 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 5 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 1 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 2 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 3 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 4 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 5 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 1 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 2 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 3 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 4 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 5 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 1 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 2 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 3 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 4 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 5 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 1 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 2 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 3 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 4 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 5 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 1 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 2 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 3 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 4 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 5 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 1 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 2 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 3 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 4 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 5 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 1 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 2 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 3 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 4 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 5 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 2 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 2 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 3 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 3 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 4 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 4 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 5 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 5 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 1 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 2 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 3 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 4 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 5 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 1 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 2 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 3 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 4 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 5 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 1 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 2 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 3 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 4 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 5 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 1 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 2 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 3 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 4 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 5 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 1 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 2 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 3 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 4 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 5 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 1 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 2 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 3 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 4 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 5 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 1 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 2 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 3 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 4 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 5 --differential-mode --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 1 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 2 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 3 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 4 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 5 --normalization "none" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 2 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 2 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 3 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 3 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 4 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 4 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 5 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 5 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 1 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 2 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 3 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 4 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 5 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 1 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 2 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 3 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 4 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 5 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 1 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 2 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 3 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 4 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 5 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 1 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 2 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 3 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 4 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 5 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 1 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 2 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 3 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 4 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 5 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 1 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 2 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 3 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 4 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 5 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 1 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 2 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 3 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 4 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 5 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 1 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 2 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 3 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 4 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 5 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 2 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 2 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 3 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 3 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 4 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 4 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 5 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 5 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 1 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 2 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 3 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 4 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 5 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 1 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 2 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 3 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 4 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 5 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 1 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 2 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 3 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 4 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 5 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 1 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 2 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 3 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 4 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 5 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 1 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 2 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 3 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 4 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 5 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 1 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 2 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 3 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 4 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 5 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 1 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 2 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 3 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 4 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 5 --differential-mode --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 1 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 2 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 3 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 4 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 5 --normalization "symlog" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 2 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 2 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 3 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 3 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 4 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 4 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 5 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 5 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 1 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 2 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 3 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 4 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 5 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 1 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 2 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 3 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 4 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 5 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 1 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 2 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 3 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 4 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 5 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 1 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 2 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 3 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 4 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 5 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 1 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 2 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 3 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 4 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 5 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 1 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 2 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 3 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 4 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 5 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 1 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 2 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 3 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 4 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 5 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 1 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 2 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 3 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 4 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 5 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 2 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 2 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 3 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 3 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 4 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 4 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 5 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 5 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 1 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 2 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 3 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 4 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 5 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 1 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 2 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 3 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 4 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 5 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 1 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 2 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 3 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 4 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 5 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 1 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 2 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 3 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 4 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 5 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 1 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 2 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 3 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 4 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 5 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 1 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 2 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 3 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 4 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 5 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 1 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 2 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 3 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 4 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 5 --differential-mode --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 1 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 2 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 3 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 4 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 5 --normalization "unitary" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 2 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 2 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 3 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 3 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 4 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 4 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 5 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "rdi" -k 5 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 1 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 2 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 3 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 4 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 5 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 1 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 2 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 3 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 4 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 2 -L 5 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 1 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 2 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 3 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 4 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 5 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 1 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 2 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 3 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 4 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 3 -L 5 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 1 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 2 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 3 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 4 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 5 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 1 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 2 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 3 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 4 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 4 -L 5 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 1 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 2 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 3 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 4 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 5 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 1 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 2 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 3 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 4 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "crdi" -k 5 -L 5 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 2 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 2 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 3 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 3 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 4 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 4 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 5 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "urdi" -k 5 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 1 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 2 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 3 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 4 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 5 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 1 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 2 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 3 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 4 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 2 -L 5 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 1 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 2 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 3 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 4 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 5 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 1 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 2 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 3 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 4 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 3 -L 5 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 1 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 2 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 3 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 4 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 5 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 1 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 2 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 3 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 4 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 4 -L 5 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 1 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 2 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 3 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 4 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 5 --differential-mode --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 1 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 2 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 3 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 4 --normalization "z-score" &
srun -N 1 -n 1 -c 128 --exclusive --unbuffered python run_all.py --mi-est "ucrdi" -k 5 -L 5 --normalization "z-score" &

wait
exit