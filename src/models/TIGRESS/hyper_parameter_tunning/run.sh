#!/bin/bash -l
#SBATCH -J bsimds_TIGRESS_hyper
#SBATCH -N 64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -p batch
#SBATCH --no-requeue

cd ~/TIGRESS/hyper_parameter_tunning
conda activate TIGRESS

for normalizer in none symlog z-score unitary
do
    for alpha in 0.1 0.2 0.5
    do
        for nstepsLARS in 3 5 8 10
        do
            for nsplit in 50 100 200 500
            do
                for scoring in area max
                do
                    srun -N 1 -n 1 --cpus-per-task 128 --exclusive --unbuffered Rscript run.R --save-folder="../results/20230405_001" --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r1.npy" --normalization-method=$normalizer --alpha=$alpha --nstepsLARS=$nstepsLARS --nsplit=$nsplit --scoring=$scoring &
                    srun -N 1 -n 1 --cpus-per-task 128 --exclusive --unbuffered Rscript run.R --save-folder="../results/20230405_001" --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r1.npy" --normalization-method=$normalizer --alpha=$alpha --nstepsLARS=$nstepsLARS --nsplit=$nsplit --scoring=$scoring --normalizeexp &
                done
            done
        done
    done
done

wait
exit	