#!/bin/bash -l
#SBATCH -J bsimds_PIDC_ts_003
#SBATCH -t 2-00:00:00
#SBATCH -N 12
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH -p batch
#SBATCH --no-requeue

cd /home/users/ttong/PIDC
conda activate PIDC

srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_10ts_003" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_10ts_003" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_10ts_003" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_10ts_003" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_10ts_003" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_10ts_003" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r3.npy" &

srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_20ts_003" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_20ts_003" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_20ts_003" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_20ts_003" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_20ts_003" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_20ts_003" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r3.npy" &

srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_30ts_003" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_30ts_003" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_30ts_003" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_30ts_003" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_30ts_003" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_30ts_003" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r3.npy" &

srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_40ts_003" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_40ts_003" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_40ts_003" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_40ts_003" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_40ts_003" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_40ts_003" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r3.npy" &

wait

srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_10ts_003" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_10ts_003" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_10ts_003" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r3.npy" &

srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_20ts_003" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_20ts_003" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_20ts_003" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r3.npy" &

srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_30ts_003" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_30ts_003" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_30ts_003" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r3.npy" &

srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_40ts_003" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_40ts_003" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_40ts_003" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r3.npy" &

wait

srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_10ts_003" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_10ts_003" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_10ts_003" --b-time-steps=10 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r3.npy" &

srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_20ts_003" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_20ts_003" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_20ts_003" --b-time-steps=20 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r3.npy" &

srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_30ts_003" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_30ts_003" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_30ts_003" --b-time-steps=30 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r3.npy" &

srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_40ts_003" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_40ts_003" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="/home/users/ttong/PIDC/results/20230516_40ts_003" --b-time-steps=40 --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims100r3.npy" &

wait
exit

