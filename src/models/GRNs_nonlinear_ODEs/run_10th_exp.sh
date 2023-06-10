#!/bin/bash -l
#SBATCH -J bsimds_NL_ODE_004
#SBATCH -t 2-00:00:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -p batch
#SBATCH --no-requeue

cd ~/GRNs_nonlinear_ODEs
conda activate GRNs_nonlinear_ODEs

srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2_n1.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2_n2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r3_n3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r3_n4.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r3_n5.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2_n1.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r3_n2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2_n3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2_n4.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2_n5.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3_n1.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3_n2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3_n3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3_n4.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3_n5.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3_n1.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3_n2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3_n3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3_n4.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3_n5.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-simulation-type="netsims" --b-suffix="_test_netsims50r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="food_webs" --b-simulation-type="netsims" --b-suffix="_test_netsims100r1.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="food_webs" --b-simulation-type="netsims" --b-suffix="_test_netsims15r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="food_webs" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="food_webs" --b-simulation-type="netsims" --b-suffix="_test_netsims50r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_coexpression_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r3.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_coexpression_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r3.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_coexpression_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r2.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_coexpression_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r2.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_regulatory_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_regulatory_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_regulatory_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_regulatory_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="intercellular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="intercellular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="intercellular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="intercellular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="landscape_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="landscape_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r3.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="landscape_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="landscape_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r2.npy" &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="man-made_organic_reaction_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="man-made_organic_reaction_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="man-made_organic_reaction_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="man-made_organic_reaction_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="reaction_networks_inside_living_organism" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="reaction_networks_inside_living_organism" --b-simulation-type="netsims" --b-suffix="_test_netsims15r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="reaction_networks_inside_living_organism" --b-simulation-type="netsims" --b-suffix="_test_netsims30r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="reaction_networks_inside_living_organism" --b-simulation-type="netsims" --b-suffix="_test_netsims50r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="social_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="social_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="social_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="social_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="vascular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="vascular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="vascular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3.npy" --b-directed &
srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="vascular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r2.npy" --b-directed &

wait
exit
