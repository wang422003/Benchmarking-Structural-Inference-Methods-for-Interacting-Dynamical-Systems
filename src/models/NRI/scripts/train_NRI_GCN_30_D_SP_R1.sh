#!/bin/bash -l
#SBATCH -J NRI_ GCN_D_springs_30_r1
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


python3 train.py --save-probs --b-network-type 'gene_coexpression_networks' --b-simulation-type 'springs' --b-suffix '30r1' --epochs 600 --batch-size 16
