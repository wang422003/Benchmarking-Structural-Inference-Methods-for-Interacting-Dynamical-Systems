#!/bin/bash -l
#SBATCH -J NRI_RNLO_D_netsims_30_N2
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=aoran.wang@uni.lu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 7
#SBATCH -G 1
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH -o %x-%j.out

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"

# Your more useful application can be started below!
module load lang/Python/3.8.6-GCCcore-10.2.0
source ~/virtualenv/py3.8.6_backup/bin/activate


python3 train.py --save-probs --b-network-type 'reaction_networks_inside_living_organism' --b-directed --b-simulation-type 'netsims' --b-suffix '30r1_n2' --epochs 2 --batch-size 16
