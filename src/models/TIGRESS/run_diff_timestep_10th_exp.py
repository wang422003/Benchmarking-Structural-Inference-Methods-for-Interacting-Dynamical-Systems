import os
import numpy as np
from sklearn.metrics import roc_auc_score
import re
import glob

list_experiments = ['/work/projects/bsimds/backup/src/simulations/brain_networks/directed/netsims/edges_test_netsims100r1.npy',
                    '/work/projects/bsimds/backup/src/simulations/brain_networks/directed/netsims/edges_test_netsims100r2.npy',
                    '/work/projects/bsimds/backup/src/simulations/brain_networks/directed/netsims/edges_test_netsims100r3.npy',
                    '/work/projects/bsimds/backup/src/simulations/brain_networks/directed/netsims/edges_test_netsims15r1.npy',
                    '/work/projects/bsimds/backup/src/simulations/brain_networks/directed/netsims/edges_test_netsims15r2.npy',
                    '/work/projects/bsimds/backup/src/simulations/brain_networks/directed/netsims/edges_test_netsims15r3.npy',
                    '/work/projects/bsimds/backup/src/simulations/brain_networks/directed/netsims/edges_test_netsims30r1.npy',
                    '/work/projects/bsimds/backup/src/simulations/brain_networks/directed/netsims/edges_test_netsims30r2.npy',
                    '/work/projects/bsimds/backup/src/simulations/brain_networks/directed/netsims/edges_test_netsims30r3.npy',
                    '/work/projects/bsimds/backup/src/simulations/brain_networks/directed/netsims/edges_test_netsims50r1.npy',
                    '/work/projects/bsimds/backup/src/simulations/brain_networks/directed/netsims/edges_test_netsims50r2.npy',
                    '/work/projects/bsimds/backup/src/simulations/brain_networks/directed/netsims/edges_test_netsims50r3.npy']

f = open("run_diff_timestep_10th_exp.sh", "w")

sbatch_header = """#!/bin/bash -l
#SBATCH -J bsimds_TIGRESS_ts_004
#SBATCH -t 2-00:00:00
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -p batch
#SBATCH --no-requeue

cd ~/TIGRESS
conda activate TIGRESS

"""

f.write(sbatch_header)

for n_ts in [10, 20, 30, 40]:
    predictions = {}
    save_folder = f"./results/20230516_{n_ts}ts_004/"

    for edge_fn in list_experiments:
        ans = np.load(edge_fn)
        num_node, rep = re.findall("netsims(\d+)(r\d).npy", edge_fn)[0]
        num_node = int(num_node)
        
        results_fns = glob.glob(f"./results/20230516_{n_ts}ts_00*/*{num_node}{rep}.csv")
        results_fns = [i for i in results_fns if "_004/" not in i]
        assert len(results_fns) == 3
        worse_score = 1
        worse_fn = ""
        for results_fn in results_fns:
            pred = np.loadtxt(results_fn, delimiter=",")
            pred_score = roc_auc_score(ans.reshape(-1), pred.reshape(-1))
            if pred_score <= worse_score:
                worse_fn = results_fn
                worse_score = pred_score
        predictions[edge_fn] = (worse_fn, worse_score)
    
    srun_format = f"srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered Rscript run.R --save-folder=\"{save_folder}\" --b-time-steps={n_ts} --b-network-type=\"brain_networks\" --b-simulation-type=\"netsims\" --b-suffix=\"%s\" &"
    for num_node in [15, 30, 50, 100]:
        worse_score = 1
        worse_fn = ""
        for k, v in predictions.items():
            if str(num_node)+"r" not in k:
                continue
            else:
                if v[1] <= worse_score:
                    worse_score = v[1]
                    worse_fn = k
        f.write(srun_format%os.path.split(worse_fn)[-1])
        f.write("\n")

f.write("\nwait\nexit\n")
f.close()
