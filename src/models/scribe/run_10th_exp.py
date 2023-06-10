import os
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
import re

import importlib
helper_func = importlib.machinery.SourceFileLoader("helper_func", "../helper_functions.py").load_module()

class fake_parser:
    def __init__(self):
        self.save_folder = ""
        self.data_path = ""
        self.b_network_type = ""
        self.b_directed = True
        self.b_simulation_type = ""
        self.b_suffix = ""

list_experiments = []

datset_directed_map = {
    "man-made_organic_reaction_networks": "directed",
    "gene_coexpression_networks": "undirected",
    "landscape_networks": "undirected",
    "brain_networks": "directed",
    "reaction_networks_inside_living_organism": "directed",
    "vascular_networks": "directed",
    "food_webs": "directed",
    "social_networks": "directed",
    "intercellular_networks": "directed",
    "gene_regulatory_networks": "directed",
    "chemical_reaction_networks_in_atmosphere": "directed"
}

parser = fake_parser()
parser.save_folder = "./results/20230422_004/"
parser.data_path = "/work/projects/bsimds/backup/src/simulations/"
parser.b_network_type = "brain_networks"
parser.b_directed = True
parser.b_simulation_type = "netsims"
parser.b_suffix = ""

srun_format = "srun -N 1 -n 1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive --unbuffered python run.py --save-folder=\"./results/20230422_004\" --b-time-steps=49 --b-network-type=\"%s\" --b-simulation-type=\"netsims\" --b-suffix=\"%s\""
sbatch_header = """#!/bin/bash -l
#SBATCH -J bsimds_scribe_004
#SBATCH -t 2-00:00:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -p batch
#SBATCH --no-requeue

cd ~/scribe/
conda activate benchmark_scribe

"""

for n_nodes in [15, 30, 50, 100]:
    parser.b_suffix = "test_netsims%dr1"%n_nodes
    list_experiments.extend(helper_func.helper_func.get_experiment_list_to_be_run(parser))

for k, v in datset_directed_map.items():
    if k != "brain_networks":
        parser.b_network_type = k
        parser.b_directed = v == "directed"
        for n_nodes in [15, 30, 50, 100]:
            parser.b_suffix = "test_netsims%dr1.npy"%(n_nodes)
            list_experiments.extend(helper_func.helper_func.get_experiment_list_to_be_run(parser))

list_experiments = sorted(list_experiments)

with open("run_10th_exp.sh", "w") as f:
    f.write(sbatch_header)
    for edge_fn in list_experiments:
        answer = (np.load(edge_fn) != 0).astype(int)
        network_type = edge_fn.replace(parser.data_path, "")
        if network_type[0] == "/":
            network_type = network_type[1:]
        network_type = network_type.split("/")[0].replace("man-made", "man_made")
        is_directed = "undirected" not in edge_fn
        suffix = re.findall(r"_netsims(.*?).npy", edge_fn)[0]

        list_file = []
        for dirpath, subdirs, files in os.walk("./results/"):
            for x in files:
                if (x.endswith(suffix+".csv") or x.endswith(suffix.replace("r1","r2")+".csv") or x.endswith(suffix.replace("r1","r3")+".csv")) and network_type in x:
                    list_file.append(os.path.abspath(os.path.join(dirpath, x)))
        assert len(list_file) == 9

        performance = {}
        min_score = 1
        min_score_fn = ""
        for fn in list_file:
            prediction = np.loadtxt(fn, delimiter=",")
            performance[fn] = roc_auc_score(answer.reshape(-1), prediction.reshape(-1))
            if min_score > performance[fn]:
                min_score = performance[fn]
                min_score_fn = fn
        
        print(min_score_fn)
        suffix = re.findall(r"_netsims(.*?).csv", min_score_fn)[0]+".npy"
        srun_print_msg = srun_format%(network_type.replace("man_made", "man-made"), suffix)
        if is_directed:
            srun_print_msg += " --b-directed"
        srun_print_msg += " &\n"
        f.write(srun_print_msg)
    f.write("\nwait\nexit\n")
