#!/bin/bash -l
#SBATCH -J NRI_train_BN_15_D_NS
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=aoran.wang@uni.lu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-12:00:00
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH -o %x-%j.out

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"

# Your more useful application can be started below!
module load lang/Python/3.8.6-GCCcore-10.2.0
source ~/virtualenv/py3.8.6_NRI/bin/activate

python3 multi_run_scripts_NS.py
