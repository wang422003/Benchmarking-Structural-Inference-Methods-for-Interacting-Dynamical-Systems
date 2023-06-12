#!/bin/bash -l
#SBATCH -J bsimds_dynGENIE3_hyper
#SBATCH -N 10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -p big
#SBATCH --no-requeue

cd ~/bsimds/src/bio_models/dynGENIE3/hypermeter_tunning
conda activate dynGENIE3

for normalizer in None symlog z-score unitary
do
	for ntree in {100..1000..100}
	do
		for max_depth in {10..100..10}
		do
			srun -N 1 -n 1 --cpus-per-task 8 --exclusive --unbuffered python test.py --b-time-steps=49 --n-trees=$ntree --max-depth=$max_depth --normalizer=$normalizer &
		done
		srun -N 1 -n 1 --cpus-per-task 8 --exclusive --unbuffered python test.py --b-time-steps=49 --n-trees=$ntree --no-max-depth --normalizer=$normalizer &
	done
done

wait
exit	
