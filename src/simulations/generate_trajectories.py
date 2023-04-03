# The original code was taken from:
# https://github.com/loeweX/AmortizedCausalDiscovery/blob/master/codebase/data/generate_dataset.py
# With modification

from synthetic_sim import MySpringSim, PyNetSim
import time
import numpy as np
import argparse
import os

def generate_springs_dataset(num_sims, length, sample_freq):
    loc_all = list()
    vel_all = list()
    edges = np.zeros(1)

    for i in range(num_sims):
        t = time.time()
        loc, vel, edges = sim.sample_trajectory(T=length,
                                                sample_freq=sample_freq)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)
        vel_all.append(vel)

    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)

    return loc_all, vel_all, edges


def generate_netsim_dataset(num_sims, length, sample_freq):

    bold_all = list()
    edges = np.zeros(1)

    for i in range(num_sims):
        t = time.time()
        bold, edges = sim.sample_trajectory(T=length,
                                                sample_freq=sample_freq)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        bold_all.append(bold)

    bold_all = np.stack(bold_all)

    return bold_all, edges

def get_substring_between_two_chars(str_, ch1='r', ch2='_'):
    return str_[::-1][str_[::-1].find(ch2) : str_[::-1].find(ch1)][::-1][:-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network-type', type=str, default='brain_networks',
                        help='What is the type of ground truth edge. (brain_networks, '
                             'chemical_reaction_networks_in_atmosphere, food_webs, gene_coexpression_networks,'
                             'gene_regulatory_networks, intercellular_networks, landscape_networks,'
                             'man-made_organic_reaction_networks, reaction_networks_inside_living_organism,'
                             'social_networks, or vascular_networks)')
    parser.add_argument('--simulation', type=str, default='springs',
                        help='What simulation to generate. (springs or netsims)')
    parser.add_argument('--num-train', type=int, default=8000,
                        help='Number of training simulations to generate.')
    parser.add_argument('--num-valid', type=int, default=2000,
                        help='Number of validation simulations to generate.')
    parser.add_argument('--num-test', type=int, default=2000,
                        help='Number of test simulations to generate.')
    parser.add_argument('--length', type=int, default=5000,
                        help='Length of trajectory.')
    parser.add_argument('--length-test', type=int, default=10000,
                        help='Length of test set trajectory.')
    parser.add_argument('--sample-freq', type=int, default=100,
                        help='How often to sample the trajectory.')
    parser.add_argument('--n-balls', type=int, default=5,
                        help='Number of balls in the simulation.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument("--save-dir", type=str, default="./data/physics_simulations", help="Where to save generated data.")
    parser.add_argument('--full-graph', action='store_true', default=False,
                        help='Whether to generate a fully-connected network.')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Whether to use the groundtruth from undirected networks.')
    parser.add_argument('--balls-limit', type=int, default=0,
                        help='Random seed.')

    args = parser.parse_args()

    edges_list = []
    curr_path = os.path.dirname(os.path.realpath(__file__))
    if not args.undirected:
        edges_dir_path = curr_path[:-11] + 'graphs/' + args.network_type + '/ready/'
    else:
        edges_dir_path = curr_path[:-11] + 'graphs/' + args.network_type + '/un_ready/'

    for file in os.listdir(edges_dir_path):
        if file.endswith('.npy'):
            # print(os.path.join(edges_dir_path, file))
            edges_list.append(os.path.join(edges_dir_path, file))

    for file in edges_list:
        # Load the adjacency matrix
        print("Working on :", file)
        edges = np.load(file)
        if args.balls_limit > 0:
            if edges.shape[0] > args.balls_limit:
                continue
        args.n_balls = edges.shape[0]
        rep = get_substring_between_two_chars(file)

        if args.simulation == 'springs':
            sim = MySpringSim(
                edges_generated=edges,
                noise_var=0.0,
                full_graph=args.full_graph
            )
            if args.full_graph:
                suffix = '_springs' + str(args.n_balls) + 'full' + rep
            else:
                suffix = '_springs' + str(args.n_balls) + 'r' + rep

            np.random.seed(args.seed)

            print(suffix)

            print("Generating {} training simulations".format(args.num_train))
            loc_train, vel_train, edges_train = generate_springs_dataset(
                args.num_train,
                args.length,
                args.sample_freq
            )

            print("Generating {} validation simulations".format(args.num_valid))
            loc_valid, vel_valid, edges_valid = generate_springs_dataset(
                args.num_valid,
                args.length,
                args.sample_freq
            )

            print("Generating {} test simulations".format(args.num_test))
            loc_test, vel_test, edges_test = generate_springs_dataset(
                args.num_test,
                args.length_test,
                args.sample_freq
            )
            if not args.undirected:
                ext_char = 'directed/'
            else:
                ext_char = 'undirected/'
            savepath = curr_path + '/' + args.network_type + '/' + ext_char + args.simulation + '/'
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            np.save(os.path.join(savepath, 'loc_train' + suffix + '.npy'), loc_train)
            np.save(os.path.join(savepath, 'vel_train' + suffix + '.npy'), vel_train)
            np.save(os.path.join(savepath, 'edges_train' + suffix + '.npy'), edges_train)

            np.save(os.path.join(savepath, 'loc_valid' + suffix + '.npy'), loc_valid)
            np.save(os.path.join(savepath, 'vel_valid' + suffix + '.npy'), vel_valid)
            np.save(os.path.join(savepath, 'edges_valid' + suffix + '.npy'), edges_valid)

            np.save(os.path.join(savepath, 'loc_test' + suffix + '.npy'), loc_test)
            np.save(os.path.join(savepath, 'vel_test' + suffix + '.npy'), vel_test)
            np.save(os.path.join(savepath, 'edges_test' + suffix + '.npy'), edges_test)

        elif args.simulation == 'netsims':
            sim = PyNetSim(
                edges_generated=edges,
                noise_var=0.0,
            )
            suffix = '_netsims' + str(args.n_balls) + 'r' + rep
            np.random.seed(args.seed)

            print(suffix)

            print("Generating {} training simulations".format(args.num_train))
            bold_train, edges_train = generate_netsim_dataset(
                args.num_train,
                args.length,
                args.sample_freq
            )

            print("Generating {} validation simulations".format(args.num_valid))
            bold_valid, edges_valid = generate_netsim_dataset(
                args.num_valid,
                args.length,
                args.sample_freq
            )

            print("Generating {} test simulations".format(args.num_test))
            bold_test, edges_test = generate_netsim_dataset(
                args.num_test,
                args.length_test,
                args.sample_freq
            )

            if not args.undirected:
                ext_char = 'directed/'
            else:
                ext_char = 'undirected/'

            savepath = curr_path + '/' + args.network_type + '/' + ext_char + args.simulation + '/'
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            np.save(os.path.join(savepath, 'bold_train' + suffix + '.npy'), bold_train)
            np.save(os.path.join(savepath, 'edges_train' + suffix + '.npy'), edges_train)

            np.save(os.path.join(savepath, 'bold_valid' + suffix + '.npy'), bold_valid)
            np.save(os.path.join(savepath, 'edges_valid' + suffix + '.npy'), edges_valid)

            np.save(os.path.join(savepath, 'bold_test' + suffix + '.npy'), bold_test)
            np.save(os.path.join(savepath, 'edges_test' + suffix + '.npy'), edges_test)
        else:
            raise ValueError('Simulation {} not implemented'.format(args.simulation))

        print("----Trajectories generation of " + file + " is finished----")


