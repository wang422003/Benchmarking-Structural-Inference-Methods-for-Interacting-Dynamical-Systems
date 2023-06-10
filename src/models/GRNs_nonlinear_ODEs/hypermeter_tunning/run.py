import os
import sys
import argparse
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from importlib import machinery
helper_functions = machinery.SourceFileLoader("helper_func", "../../helper_functions.py").load_module()
xgbgrn = machinery.SourceFileLoader("xgbgrn", "../xgbgrn.py").load_module()

feature_dim = 0

parser = argparse.ArgumentParser(description='Run hyperparameters test on GRNs_nonlinear_ODEs')
parser.add_argument('--data-path', type=str, default="/work/projects/bsimds/backup/src/simulations/chemical_reaction_networks_in_atmosphere/directed/netsims/edges_test_netsims15r1.npy",
                    help="The file where data are stored.")
parser.add_argument('--b-time-steps', type=int, default=49,
                    help='Portion of time series in data to be used in benchmarking.')
parser.add_argument('--normalizer', type=str, default="none",
                    help='Normalizer can choose from "none", "symlog", "unitary", "z-score"')
parser.add_argument('--n-estimators', type=int, required=True,
                    help='Number of estimators in XGBoost.')
parser.add_argument('--learning-rate', type=float, required=True,
                    help='Learning rate in XGBoost.')
parser.add_argument('--max-depth', type=int, required=True,
                    help='Maximum depth of trees in XGBoost.')
parser.add_argument('--subsample', type=float, required=True,
                    help='Portion of sample in each iteration when fitting XGBoost.')                    
parser.add_argument('--alpha', type=float, required=True,
                    help='Alpha value used while fitting XGBoost.')                    
args = parser.parse_args()

if args.normalizer == "none": normalizer = None
elif args.normalizer == "symlog": normalizer = helper_functions.helper_func.sym_log_normalizer
elif args.normalizer == "unitary": normalizer = helper_functions.helper_func.unitary_normalizer
elif args.normalizer == "z-score": normalizer = helper_functions.helper_func.z_score_normalizer
else: raise Exception('Normalizer must be chosen from "none", "symlog", "unitary", "z-score"')

exp_data = helper_functions.helper_func.read_exp_data(args.data_path, raw = True)
exp_data["data"] = [exp_data["data"][i,:args.b_time_steps,feature_dim,:] for i in range(exp_data["data"].shape[0])]
exp_data["data"] = [arr if normalizer is None else normalizer(arr) for arr in exp_data["data"]]
time_points = [np.arange(exp_data["data"][0].shape[0]) for i in exp_data["data"]]

xgb_kwargs = {
    "importance_type": "weight",
    "n_jobs": -1,
    "validate_parameters": True,
    "tree_method": "hist",
    "n_estimators": args.n_estimators,
    "learning_rate": args.learning_rate,
    "max_depth": args.max_depth,
    "subsample": args.subsample,
    "alpha": args.alpha
}

n_genes = exp_data["data"][0].shape[1]
gene_names = ['G'+str(i+1) for i in range(n_genes)]
regulators = gene_names.copy()

VIM = xgbgrn.xgbgrn().get_importances(exp_data["data"], time_points, gene_names=gene_names, regulators=regulators, param=xgb_kwargs)

np.savetxt("./inferred_network_%s_%d_%f_%d_%f_%f.csv"%(args.normalizer, args.n_estimators, args.learning_rate, args.max_depth, args.subsample, args.alpha), VIM, delimiter = ",")
