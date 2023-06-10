from dynGENIE3 import dynGENIE3
import numpy as np
import os
import sys
import argparse

import warnings
warnings.filterwarnings("ignore")

import importlib
helper_functions = importlib.machinery.SourceFileLoader("helper_func", "../../helper_functions.py").load_module()

feature_dim = 0

parser = argparse.ArgumentParser(description='Run hyperparameters test on dynGENIE3')
parser.add_argument('--data-path', type=str, default="/shared/projects/BSIMDS/src/data/Local_Project/src/simulations/chemical_reaction_networks_in_atmosphere/directed/netsims/edges_test_netsims15r1.npy",
                    help="The file where data are stored.")
parser.add_argument('--b-time-steps', type=int, default=49,
                    help='Portion of time series in data to be used in benchmarking.')
parser.add_argument('--normalizer', type=str, default="None",
                    help='Normalizer can choose from "None", "symlog", "unitary", "z-score"')
parser.add_argument('--n-trees', type=int, default=50,
                    help='Number of trees in random forest.')
parser.add_argument('--max-depth', type=int, default=10,
                    help='Maximum depth of trees in random forest.')
parser.add_argument('--no-max-depth', action="store_true",
                    help='Maximum depth of trees in random forest.')
args = parser.parse_args()

if args.normalizer == "None": normalizer = None
elif args.normalizer == "symlog": normalizer = helper_functions.helper_func.sym_log_normalizer
elif args.normalizer == "unitary": normalizer = helper_functions.helper_func.unitary_normalizer
elif args.normalizer == "z-score": normalizer = helper_functions.helper_func.z_score_normalizer
else: raise Exception('Normalizer must be chosen from "None", "log", "unitary", "z-score"')

exp_data = helper_functions.helper_func.read_exp_data(args.data_path, raw = True)
exp_data["data"] = [exp_data["data"][i,:args.b_time_steps,feature_dim,:] for i in range(exp_data["data"].shape[0])]
exp_data["data"] = [arr if normalizer is None else normalizer(arr) for arr in exp_data["data"]]
time_points = [np.arange(exp_data["data"][0].shape[0]) for i in exp_data["data"]]

if args.no_max_depth == False:
    (VIM, _, _, _, _) = dynGENIE3(exp_data["data"], time_points, nthreads=os.cpu_count(), ntrees=args.n_trees, max_depth=args.max_depth, K = "all")
    np.savetxt("./inferred_network_norm_%s_nT%d_maxD%d.csv"%(args.normalizer, args.n_trees, args.max_depth), VIM, delimiter = ",")
else:
    (VIM, _, _, _, _) = dynGENIE3(exp_data["data"], time_points, nthreads=os.cpu_count(), ntrees=args.n_trees, max_depth=None, K = "all")
    np.savetxt("./inferred_network_norm_%s_nT%d_maxD%s.csv"%(args.normalizer, args.n_trees, "_None_"), VIM, delimiter = ",")
