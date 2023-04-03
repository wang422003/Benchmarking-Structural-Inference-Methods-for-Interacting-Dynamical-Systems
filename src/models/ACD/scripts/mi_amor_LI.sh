#!/bin/bash -l
#SBATCH -J MI_amor_LI_CUDA
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
module load lang/Python/3.7.4-GCCcore-8.3.0
source ~/virtualenv/py3.7.4_NRI/bin/activate


python ~/amortized/codebase/mi_cal_iris.py --suffix='LI' --exp-folder=x --sample-epochs
