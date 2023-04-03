import numpy as np
from mi_calculation import *
import argparse
import pickle
import torch


def num_nodes(data_suffix='LI'):
    if data_suffix == 'LI':
        return 7
    elif data_suffix == 'LL':
        return 18
    elif data_suffix == 'CY':
        return 6
    elif data_suffix == 'BF':
        return 7
    elif data_suffix == 'TF':
        return 8
    elif data_suffix == 'BF-CV':
        return 10
    else:
        raise ValueError("Check the suffix of the dataset!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='LI',
                        help='Suffix for training data (e.g. "LI").')
    parser.add_argument('--exp-folder', type=str, default='LI-it-Emlp-Dmlp-exp2021-12-04T18:06:18.518728',
                        help='The folder of the experiment. The path should point to a folder on HPC servers.')
    parser.add_argument('--off-diag', action='store_true', default=False,
                        help='Discard diagonal elements.')
    parser.add_argument('--sample-epochs', action='store_true', default=False,
                        help='Sample .npy files according to their epochs.')
    args = parser.parse_args()
    args.probs_folder = 'logs/' + args.exp_folder + '/probs/'
    args.num_nodes = num_nodes(args.suffix)
    print(args)

    mi_calculation_name_order(
        probs_folder_path=args.probs_folder,
        num_nodes=args.num_nodes,
        suffix=args.suffix,
        off_d=args.off_diag,
        sample_epochs=args.sample_epochs
    )

    print("Mutual Information Calculation Finished!")