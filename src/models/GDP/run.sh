#!/bin/bash -l
#SBATCH -J bsimds_GDP_001_hyper
#SBATCH --mail-type end,fail
#SBATCH --mail-user tszpan.tong@uni.lu
#SBATCH -N 50
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 16
#SBATCH --time=2-0:00:00
#SBATCH -p gpu
#SBATCH -o "/project/scratch/p200352/bsimds/src/models/GDP/results/slurm-%j.out"
#SBATCH -e "/project/scratch/p200352/bsimds/src/models/GDP/results/slurm-%j.err"
#SBATCH -A p200352
#SBATCH -q default

cd /project/scratch/p200352/bsimds/src/models/GDP
conda activate playground
module load env/release/2023.1
module load cuDNN/8.9.2.26-CUDA-12.2.0

save_folder="/project/scratch/p200352/bsimds/src/models/GDP/results/20240521_001_hyper"
mkdir -p $save_folder
# export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=10

# srun -G 1 -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python train_DYGR.py --batch_size 256 --save-folder=$save_folder --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-directed --b-simulation-type="netsims" --b-suffix="15r1" --suffix "netsims" &
# srun -G 1 -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python train_DYGR.py --batch_size 64 --save-folder=$save_folder --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-directed --b-simulation-type="netsims" --b-suffix="30r1" --suffix "netsims" &
# srun -G 1 -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python train_DYGR.py --batch_size 16 --save-folder=$save_folder --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-directed --b-simulation-type="netsims" --b-suffix="50r1" --suffix "netsims" &
# srun -G 1 -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python train_DYGR.py --batch_size 8 --save-folder=$save_folder --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-directed --b-simulation-type="netsims" --b-suffix="100r1" --suffix "netsims" &

counter=0
for num_layers in 3 2 1
do
    for heads in 3 2 1
    do
        for K in 1 2 3 4 5
        do
            for filter in cheby power
            do
                for dropout in 0.0 0.3 0.5
                do
                    if [ $counter -ge 200 ]
                    then
                        srun -G 1 -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python train_DYGR.py --batch_size 256 --num_epoch 200 --save-folder=$save_folder --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-directed --b-simulation-type="springs" --b-suffix="15r1" --suffix="springs" --seed=1 --dropout=$dropout --filter=$filter --K=$K --num-layers=$num_layers --heads=$heads --Tstep=2 &
                        srun -G 1 -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python train_DYGR.py --batch_size 256 --num_epoch 200 --save-folder=$save_folder --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-directed --b-simulation-type="springs" --b-suffix="15r2" --suffix="springs" --seed=1 --dropout=$dropout --filter=$filter --K=$K --num-layers=$num_layers --heads=$heads --Tstep=2 &
                        srun -G 1 -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python train_DYGR.py --batch_size 256 --num_epoch 200 --save-folder=$save_folder --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-directed --b-simulation-type="springs" --b-suffix="15r3" --suffix="springs" --seed=1 --dropout=$dropout --filter=$filter --K=$K --num-layers=$num_layers --heads=$heads --Tstep=2 &
                        srun -G 1 -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python train_DYGR.py --batch_size 256 --num_epoch 200 --save-folder=$save_folder --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-directed --b-simulation-type="springs" --b-suffix="15r1" --suffix="springs" --seed=2 --dropout=$dropout --filter=$filter --K=$K --num-layers=$num_layers --heads=$heads --Tstep=2 &
                        srun -G 1 -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python train_DYGR.py --batch_size 256 --num_epoch 200 --save-folder=$save_folder --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-directed --b-simulation-type="springs" --b-suffix="15r2" --suffix="springs" --seed=2 --dropout=$dropout --filter=$filter --K=$K --num-layers=$num_layers --heads=$heads --Tstep=2 &
                    fi
                    counter=$((counter+5))                    
                done
            done
        done
    done
done

wait
exit
