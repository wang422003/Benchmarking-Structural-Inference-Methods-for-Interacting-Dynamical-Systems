import pandas as pd
import numpy as np
import argparse
import os

class helper_func:
    def read_exp_data(edge_file_path, raw = False):
        edge_file_path = os.path.abspath(edge_file_path)
        current_folder = os.path.dirname(edge_file_path)
        if current_folder == '':
            current_folder = "."
        edge_file_name = os.path.basename(edge_file_path)
        data_file = [os.path.abspath(os.path.join(current_folder, fn)) for fn in os.listdir(current_folder) if edge_file_name.replace("edges_", "") in fn and edge_file_name not in fn]
        # all .npy files have shape (n_trajectory, n_timestep, n_feature, n_node)
        out = {"data" : None, "ground_truth" : None, "current_folder" : current_folder, "edge_file_name" : edge_file_name}
        for file in data_file:
            data = np.load(file)
            if raw == False:
                data = data.reshape((-1,data.shape[3])).T
            if out["data"] is None:
                out["data"] = data.copy()
            else:
                out["data"] = np.append(out["data"], data, axis = 0)
        
        out["ground_truth"] = np.load(edge_file_path)
        return(out)

    def get_parser():
        parser = argparse.ArgumentParser(description='Run biological models and configuration.')
        parser.add_argument('--data-path', type=str, default="/work/projects/bsimds/backup/src/simulations/",
                            help="The folder where data are stored.")
        parser.add_argument('--save-folder', type=str, required=True,
                            help="The folder where resulting adjacency matrixes are stored.")
        parser.add_argument('--b-portion', type=float, default=1.0,
                            help='Portion of data to be used in benchmarking.')
        parser.add_argument('--b-time-steps', type=int, default=49,
                            help='Portion of time series in data to be used in benchmarking.')
        parser.add_argument('--b-shuffle', action='store_true', default=False,
                            help='Shuffle the data for benchmarking?')
        parser.add_argument('--b-network-type', type=str, default='',
                            help='What is the network type of the graph.')
        parser.add_argument('--b-directed', action='store_true', default=False,
                            help='Default choose trajectories from undirected graphs.')
        parser.add_argument('--b-simulation-type', type=str, default='',
                            help='Either springs or netsims.')
        parser.add_argument('--b-suffix', type=str, default='',
            help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1.'
                 ' Or "50r1" for 50 nodes, rep 1 and noise free.')
        parser.add_argument('--pct-cpu', type=float, default=1.0,
                          help='Percentage of number of CPUs to be used.')
        args = parser.parse_args()
        return(args)

    def get_experiment_list_to_be_run(parser):
        if parser.save_folder == "":
            raise Exception("No save_folder provided.")
        else:
            os.makedirs(parser.save_folder, exist_ok=True)
        
        parser.data_path = os.path.abspath(parser.data_path)
        parser.save_folder = os.path.abspath(parser.save_folder)

        list_file = []
        for dirpath, subdirs, files in os.walk(parser.data_path):
            for x in files:
                if x.endswith(".npy"):
                    list_file.append(os.path.abspath(os.path.join(dirpath, x)))
        list_file = pd.Series([i for i in list_file if i.count(os.path.sep) == parser.data_path.count(os.path.sep) + 4])
        list_ground_truth = list_file[list_file.str.contains("edges")]
        
        if parser.b_network_type != "":
            list_ground_truth = list_ground_truth[(list_ground_truth.str.split(os.path.sep, expand=True) == parser.b_network_type).sum(axis = 1) != 0]
        if parser.b_directed == False:
            list_ground_truth = list_ground_truth[(list_ground_truth.str.split(os.path.sep, expand=True) == "undirected").sum(axis = 1) != 0]
        else:
            list_ground_truth = list_ground_truth[(list_ground_truth.str.split(os.path.sep, expand=True) == "directed").sum(axis = 1) != 0]
        if parser.b_simulation_type != "":
            list_ground_truth = list_ground_truth[(list_ground_truth.str.split(os.path.sep, expand=True) == parser.b_simulation_type).sum(axis = 1) != 0]
        if parser.b_suffix != "":
            list_ground_truth = list_ground_truth[list_ground_truth.apply(lambda x: parser.b_suffix in os.path.basename(x))]
        return(list_ground_truth.to_list())

    def get_result_file_names(list_experiements, parser):
        list_experiements = pd.Series(list_experiements)
        out_file_names = list_experiements.str.replace(parser.data_path, "")
        out_file_names = out_file_names.str.replace(".npy", ".csv")
        out_file_names = out_file_names.str.replace(r"[\-/\\]", "_", regex = True)
        out_file_names = out_file_names.str.replace("edges_", "")
        is_duplicated_file = out_file_names.isin(os.listdir(parser.save_folder))
        out_file_names = out_file_names[~is_duplicated_file].to_list()
        list_experiements = list_experiements[~is_duplicated_file].to_list()

        for idx, fn in enumerate(out_file_names):
            while fn[0] == "_":
                fn = fn[1:]
            out_file_names[idx] = fn
        return(out_file_names, list_experiements)

    def sym_log_normalizer(arr):
        return(np.log1p(np.abs(arr)) * np.sign(arr))

    def unitary_normalizer(arr, axis = -1, **kwarg):
        return(arr / np.linalg.norm(arr, axis = axis, keepdims = True, **kwarg))

    def z_score_normalizer(arr, axis = -1):
        return((arr - arr.mean(axis = axis, keepdims = True) / arr.std(axis = axis, keepdims = True)))
