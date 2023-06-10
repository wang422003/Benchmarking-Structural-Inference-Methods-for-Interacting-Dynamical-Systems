#!/bin/bash -l
#SBATCH -J bsimds_ppcor_ts_001
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH -p batch
#SBATCH --no-requeue

cd /home/users/ttong/ppcor
conda activate ppcor

srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_10ts_001" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_10ts_001" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_10ts_001" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_10ts_001" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_10ts_001" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_10ts_001" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_10ts_001" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_10ts_001" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_10ts_001" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_10ts_001" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_10ts_001" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_10ts_001" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r3.npy" &

srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_20ts_001" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_20ts_001" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_20ts_001" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_20ts_001" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_20ts_001" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_20ts_001" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_20ts_001" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_20ts_001" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_20ts_001" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_20ts_001" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_20ts_001" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_20ts_001" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r3.npy" &

srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_30ts_001" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_30ts_001" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_30ts_001" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_30ts_001" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_30ts_001" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_30ts_001" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_30ts_001" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_30ts_001" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_30ts_001" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_30ts_001" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_30ts_001" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_30ts_001" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r3.npy" &

srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_40ts_001" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_40ts_001" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_40ts_001" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_40ts_001" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_40ts_001" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_40ts_001" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_40ts_001" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_40ts_001" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_40ts_001" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_40ts_001" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_40ts_001" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript run.R --save-folder="/home/users/ttong/ppcor/results/20230516_40ts_001" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r3.npy" &


wait
exit

