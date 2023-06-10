import numpy as np
import os
import sys
import copy
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import importlib
helper_functions = importlib.machinery.SourceFileLoader("helper_func", "../helper_functions.py").load_module()
causal_network = importlib.machinery.SourceFileLoader("func", "causal_network.py").load_module()

if __name__ == '__main__':
    n_jobs = os.cpu_count()
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
                exp_data["data"] = helper_functions.helper_func.unitary_normalizer(exp_data["data"])
                exp_data["data"] = exp_data["data"][:,:parser.b_time_steps,0,:]
                indexs = pd.MultiIndex.from_product([np.arange(exp_data["data"].shape[0]), np.arange(exp_data["data"].shape[1]), ["G%d"%i for i in range(exp_data["data"].shape[2])]], names = ["RUN_ID", "Time","GENE_ID"])
                exp_data["data"] = pd.Series(exp_data["data"].reshape(-1), index = indexs).unstack(level = 1).swaplevel("RUN_ID", "GENE_ID")

                model = causal_network.causal_model()
                model.expression = exp_data["data"]
                model.expression_raw = copy.deepcopy(exp_data["data"])
                model.node_ids = model.expression.index.levels[0]
                model.run_ids = model.expression.index.levels[1]
                model.expression_concatenated = pd.concat([pd.DataFrame(model.expression.loc[node_id].values.reshape(1, -1), index=[node_id]) for node_id in model.node_ids])
                model.k = 2
                net = model.rdi([1], number_of_processes = n_jobs, uniformization = True, differential_mode = False)
                net["MAX"].replace([np.inf, -np.inf], np.nan).fillna(0).to_csv(os.path.join(parser.save_folder, out_file_name), index = False, header = False)
