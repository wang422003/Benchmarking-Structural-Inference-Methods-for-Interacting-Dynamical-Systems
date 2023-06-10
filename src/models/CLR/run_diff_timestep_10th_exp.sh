#!/bin/bash -l
#SBATCH -J bsimds_CLR_ts_004
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH -p batch
#SBATCH --no-requeue

cd ~/CLR
conda activate CLR

srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_10ts_004/" --b-time-steps=10 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims15r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_10ts_004/" --b-time-steps=10 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims30r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_10ts_004/" --b-time-steps=10 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims50r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_10ts_004/" --b-time-steps=10 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims100r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_20ts_004/" --b-time-steps=20 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims15r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_20ts_004/" --b-time-steps=20 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims30r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_20ts_004/" --b-time-steps=20 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims50r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_20ts_004/" --b-time-steps=20 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims100r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_30ts_004/" --b-time-steps=30 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims15r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_30ts_004/" --b-time-steps=30 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims30r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_30ts_004/" --b-time-steps=30 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims50r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_30ts_004/" --b-time-steps=30 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims100r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_40ts_004/" --b-time-steps=40 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims15r1.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_40ts_004/" --b-time-steps=40 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims30r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_40ts_004/" --b-time-steps=40 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims50r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered Rscript test.R --save-folder="./results/20230516_40ts_004/" --b-time-steps=40 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims100r2.npy" &

wait
exit
