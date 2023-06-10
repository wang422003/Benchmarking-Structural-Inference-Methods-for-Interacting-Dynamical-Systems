#!/bin/bash -l
#SBATCH -J bsimds_PIDC_004
#SBATCH -N 1
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH -p bigmem
#SBATCH --no-requeue

cd ~/PIDC
#export PATH=$PATH:~/bsimds/src/bio_models/PIDC/julia-1.6.7/bin
#module load lang/Julia/1.8.5-linux-x86_64
conda activate PIDC

srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r3_n1.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2_n2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2_n3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2_n4.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2_n5.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2_n1.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2_n2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2_n3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2_n4.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2_n5.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3_n1.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3_n2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3_n3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3_n4.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3_n5.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3_n1.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3_n2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3_n3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3_n4.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="brain_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3_n5.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-simulation-type="netsims" --b-suffix="_test_netsims100r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-simulation-type="netsims" --b-suffix="_test_netsims15r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="chemical_reaction_networks_in_atmosphere" --b-simulation-type="netsims" --b-suffix="_test_netsims50r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="food_webs" --b-simulation-type="netsims" --b-suffix="_test_netsims100r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="food_webs" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="food_webs" --b-simulation-type="netsims" --b-suffix="_test_netsims30r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="food_webs" --b-simulation-type="netsims" --b-suffix="_test_netsims50r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_coexpression_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_coexpression_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_coexpression_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_coexpression_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_regulatory_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_regulatory_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_regulatory_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="gene_regulatory_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="intercellular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="intercellular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="intercellular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="intercellular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="landscape_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="landscape_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="landscape_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r2.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="landscape_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3.npy" &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="man-made_organic_reaction_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="man-made_organic_reaction_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="man-made_organic_reaction_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="man-made_organic_reaction_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="reaction_networks_inside_living_organism" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="reaction_networks_inside_living_organism" --b-simulation-type="netsims" --b-suffix="_test_netsims15r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="reaction_networks_inside_living_organism" --b-simulation-type="netsims" --b-suffix="_test_netsims30r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="reaction_networks_inside_living_organism" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="social_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="social_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="social_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="social_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="vascular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims100r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="vascular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims15r2.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="vascular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims30r3.npy" --b-directed &
srun -N 1 -n 1 --exclusive --unbuffered julia -- run.jl --save-folder="./results/20230422_004" --b-time-steps=49 --b-network-type="vascular_networks" --b-simulation-type="netsims" --b-suffix="_test_netsims50r2.npy" --b-directed &

wait
exit
