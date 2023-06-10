import os
import sys
import argparse
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from importlib import machinery
helper_functions = machinery.SourceFileLoader("helper_func", "../helper_functions.py").load_module()
xgbgrn = machinery.SourceFileLoader("xgbgrn", "./xgbgrn.py").load_module()

parser = helper_functions.helper_func.get_parser()
list_experiment = helper_functions.helper_func.get_experiment_list_to_be_run(parser)

if len(list_experiment) == 0: 
    print("There is no experiments to be run. This may because you have specified a wrong data-path or wrong filter name.")
    exit()

out_file_names, list_experiment = helper_functions.helper_func.get_result_file_names(list_experiment, parser)

if len(out_file_names) != 0:
    for experiment_id in range(len(out_file_names)):
        out_file_name = out_file_names[experiment_id]
        if out_file_name not in os.listdir(parser.save_folder):
            exp_data = helper_functions.helper_func.read_exp_data(list_experiment[experiment_id], raw = True)
            exp_data["data"] = [exp_data["data"][i,:parser.b_time_steps,0,:] for i in range(exp_data["data"].shape[0])]
            exp_data["data"] = [helper_functions.helper_func.unitary_normalizer(arr) for arr in exp_data["data"]]
            time_points = [np.arange(exp_data["data"][0].shape[0]) for i in exp_data["data"]]
            
            xgb_kwargs = {
                "importance_type": "weight",
                "n_jobs": -1,
                "validate_parameters": True,
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 0.6,
                "alpha": 0.02,
            }

            n_genes = exp_data["data"][0].shape[1]
            gene_names = ['G'+str(i+1) for i in range(n_genes)]
            regulators = gene_names.copy()

            VIM = xgbgrn.xgbgrn().get_importances(exp_data["data"], time_points, gene_names=gene_names, regulators=regulators, param=xgb_kwargs)
            np.savetxt(os.path.join(parser.save_folder, out_file_name), VIM, delimiter = ",")
