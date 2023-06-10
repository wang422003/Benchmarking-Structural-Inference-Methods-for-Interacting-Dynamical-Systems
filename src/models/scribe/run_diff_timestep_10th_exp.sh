#!/bin/bash -l
#SBATCH -J bsimds_scribe_ts_004
#SBATCH --mail-type start,end,fail
#SBATCH --mail-user tszpan.tong@uni.lu
#SBATCH -t 2-00:00:00
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -p batch
#SBATCH --no-requeue

cd /home/users/ttong/scribe
conda activate benchmark_scribe

srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_10ts_004/" --b-time-steps=10 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims15r1.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_10ts_004/" --b-time-steps=10 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims30r3.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_10ts_004/" --b-time-steps=10 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims50r3.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_10ts_004/" --b-time-steps=10 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims100r2.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_20ts_004/" --b-time-steps=20 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims15r1.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_20ts_004/" --b-time-steps=20 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims30r3.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_20ts_004/" --b-time-steps=20 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims50r2.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_20ts_004/" --b-time-steps=20 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims100r1.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_30ts_004/" --b-time-steps=30 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims15r1.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_30ts_004/" --b-time-steps=30 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims30r3.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_30ts_004/" --b-time-steps=30 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims50r2.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_30ts_004/" --b-time-steps=30 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims100r1.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_40ts_004/" --b-time-steps=40 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims15r1.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_40ts_004/" --b-time-steps=40 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims30r3.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_40ts_004/" --b-time-steps=40 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims50r2.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230516_40ts_004/" --b-time-steps=40 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="edges_test_netsims100r1.npy" &

wait
exit
