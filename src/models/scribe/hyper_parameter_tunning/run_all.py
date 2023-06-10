import numpy as np
import copy
import pandas as pd
import os
import argparse

import warnings
warnings.filterwarnings("ignore")

import importlib
causal_network = importlib.machinery.SourceFileLoader("func", "../causal_network.py").load_module()
helper_functions = importlib.machinery.SourceFileLoader("helper_func", "../../helper_functions.py").load_module()

parser = argparse.ArgumentParser(description='Run hyper-parameter tunning for scribe.')
parser.add_argument('--mi-est', type=str, required=True)
parser.add_argument('-k', type=int, required=True)
parser.add_argument('--differential-mode', action = "store_true")
parser.add_argument('-L', type=int)
parser.add_argument('--normalization', type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    data_path = r"/work/projects/bsimds/backup/src/simulations/chemical_reaction_networks_in_atmosphere/directed/netsims/bold_test_netsims15r1.npy"
    feature_dim = 0
    n_jobs = os.cpu_count()
    print(args)

    if args.normalization == "none": normalizer = None
    elif args.normalization == "symlog": normalizer = helper_functions.helper_func.sym_log_normalizer
    elif args.normalization == "unitary": normalizer = helper_functions.helper_func.unitary_normalizer
    elif args.normalization == "z-score": normalizer = helper_functions.helper_func.z_score_normalizer
    else: raise Exception('Normalization must be chosen from "none", "log", "unitary", "z-score"')

    data = np.load(data_path)[:,:49,feature_dim,:]

    if args.normalization != "none":
        data = normalizer(data)

    indexs = pd.MultiIndex.from_product([np.arange(data.shape[0]), np.arange(data.shape[1]), ["G%d"%i for i in range(data.shape[2])]], names = ["RUN_ID", "Time","GENE_ID"])
    data = pd.Series(data.reshape(-1), index = indexs).unstack(level = 1).swaplevel("RUN_ID", "GENE_ID")

    def save_rdi_with_lags(model, uniformization = False, differential_mode = False, save_name_prefix = ""):
        net = model.rdi(np.arange(1, 21), number_of_processes = n_jobs, uniformization = uniformization, differential_mode = differential_mode)
        print(f"{save_name_prefix}_20.csv")
        net["MAX"].replace([np.inf, -np.inf], np.nan).fillna(0).to_csv(f"{save_name_prefix}_20.csv", index = False, header = False)
        previous_lag = 20
        for lag in np.arange(1, 20)[::-1]:
            del model.rdi_results[previous_lag]
            net = model.extract_max_rdi_value_delay()[0]
            print(f"{save_name_prefix}_{lag}.csv")
            net.replace([np.inf, -np.inf], np.nan).fillna(0).to_csv(f"{save_name_prefix}_{lag}.csv", index = False, header = False)
            previous_lag = lag
        return()

    model = causal_network.causal_model()
    model.expression = copy.deepcopy(data)
    model.expression_raw = copy.deepcopy(data)
    model.node_ids = model.expression.index.levels[0]
    model.run_ids = model.expression.index.levels[1]
    model.expression_concatenated = pd.concat([pd.DataFrame(model.expression.loc[node_id].values.reshape(1, -1), index=[node_id]) for node_id in model.node_ids])
    model.k = args.k

    if "c" in args.mi_est:
        save_name = f"./inferred_network_{args.normalization}_{args.mi_est}_{args.differential_mode}_{args.k}_{args.L}_none_.csv"
        print(save_name)
        if save_name in os.listdir():
            exit
        net = model.crdi(L=args.L, number_of_processes=n_jobs, uniformization=args.mi_est.startswith("u"), differential_mode=args.differential_mode)
        net.replace([np.inf, -np.inf], np.nan).fillna(0).to_csv(save_name, index = False, header = False)
    else:
        save_name_prefix = f"./inferred_network_{args.normalization}_{args.mi_est}_{args.differential_mode}_{args.k}_none"
        print(save_name_prefix)
        if sum([save_name_prefix in i for i in os.listdir()]) == 20:
            exit
        save_rdi_with_lags(model, uniformization = args.mi_est.startswith("u"), differential_mode = args.differential_mode, save_name_prefix = save_name_prefix)