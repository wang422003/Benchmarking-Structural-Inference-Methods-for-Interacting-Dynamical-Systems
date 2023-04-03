#!/bin/bash -l
#SBATCH -J Generate_data_for_amortized
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=aoran.wang@uni.lu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 7
#SBATCH --time=2-00:00:00
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH -o %x-%j.out

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"

# Your more useful application can be started below!
module load lang/Python/3.7.4-GCCcore-8.3.0
source ~/virtualenv/py3.7.4_amortized/bin/activate

python ~/amortized/codebase/data/generate_dataset.py
