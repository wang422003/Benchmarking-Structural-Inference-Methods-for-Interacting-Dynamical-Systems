from dynGENIE3 import dynGENIE3
import numpy as np
import os
import sys

import importlib
helper_functions = importlib.machinery.SourceFileLoader("helper_func", "../helper_functions.py").load_module()

if __name__ == '__main__':
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
                exp_data["data"] = [helper_functions.helper_func.z_score_normalizer(arr) for arr in exp_data["data"]]
                time_points = [np.arange(exp_data["data"][0].shape[0]) for i in exp_data["data"]]
                (VIM, _, _, _, _) = dynGENIE3(exp_data["data"], time_points, nthreads=os.cpu_count(), K = "all", max_depth = 90, ntrees = 700)
                VIM = VIM.T
                np.savetxt(os.path.join(parser.save_folder, out_file_name), VIM, delimiter = ",")
