#!/bin/bash -l
#SBATCH -J bsimds_GRN_nonlinear_ODEs_hyper
#SBATCH -N 48
#SBATCH -t 1-00:00:00
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH -p batch
#SBATCH --no-requeue

cd /home/users/ttong/GRNs_nonlinear_ODEs/hypermeter_tunning
conda activate GRNs_nonlinear_ODEs

for normalizer in none symlog z-score unitary
do
	for n_estimators in 100 200 500 1000
    do
        for learning_rate in 0.01 0.02 0.05 0.1
        do
            for max_depth in 0 3 5 6 8 10
            do
                for subsample in 0.6 0.8 1
                do
                    for alpha in 0 0.01 0.02 0.05
                    do
                        srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --b-time-steps=49 --n-estimators=$n_estimators --max-depth=$max_depth --normalizer=$normalizer --learning-rate=$learning_rate --subsample=$subsample --alpha=$alpha &
                    done
                done
            done
        done
    done
done

wait
exit	
