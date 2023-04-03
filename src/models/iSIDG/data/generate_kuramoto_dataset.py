# The original code was taken from:
# https://github.com/loeweX/AmortizedCausalDiscovery/blob/master/codebase/data/generate_ODE_dataset.py
# With modification

import numpy as np
import os
import time
import argparse

import kuramoto


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=8000, help="Number of training simulations to generate.")
    parser.add_argument("--num-valid", type=int, default=2000, help="Number of validation simulations to generate.")
    parser.add_argument("--num-test", type=int, default=2000, help="Number of test simulations to generate.")
    parser.add_argument("--length", type=int, default=5000, help="Length of trajectory.")
    parser.add_argument("--length-test", type=int, default=10000, help="Length of test set trajectory.")
    parser.add_argument("--num-atoms", type=int, default=5,
                        help="Number of atoms (aka time-series) in the simulation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--ode-type", type=str, default="kuramoto", help="Which ODE to use [kuramoto]")
    parser.add_argument('--sample-freq', type=int, default=100,
                        help='How often to sample the trajectory.')
    parser.add_argument('--interaction_strength', type=int, default=1,
                        help='Strength of Interactions between particles')
    parser.add_argument("--undirected", action="store_true", default=False, help="Have symmetric connections")
    parser.add_argument("--save-dir", type=str, default="./data/physics_simulations", help="Where to save generated data.")
    parser.add_argument("--n-save-small", type=int, default=100,
                        help="Number of training sequences to save separately.")
    args = parser.parse_args()
    print(args)
    return args


def generate_dataset(num_sims, length, sample_freq):
    num_sims = num_sims
    num_timesteps = int((length / float(sample_freq)) - 1)

    t0, t1, dt = 0, int((length / float(sample_freq)) / 10), 0.01
    T = np.arange(t0, t1, dt)

    sim_data_all = []
    # edges_all = []

    edges = np.random.choice(2, size=(args.num_atoms, args.num_atoms), p=[0.5, 0.5])
    edges = np.tril(edges) + np.tril(edges, -1).T
    for i in range(num_sims):
        t = time.time()

        if args.ode_type == "kuramoto":
            sim_data, edges = kuramoto.simulate_kuramoto(
                args.num_atoms, edges, num_timesteps, T, dt, args.undirected
            )
            assert sim_data.shape[2] == 4
        else:
            raise Exception("Invalid args.ode_type")

        sim_data_all.append(sim_data)
        # edges_all.append(edges)

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))

    data_all = np.array(sim_data_all, dtype=np.float32)
    # edges_all = np.array(edges_all, dtype=np.int64)

    return data_all, edges


if __name__ == "__main__":

    args = parse_args()
    np.random.seed(args.seed)

    suffix = "_" + args.ode_type

    # suffix += str(args.num_atoms)

    if args.undirected:
        suffix += "undir"

    if args.interaction_strength != 1:
        suffix += "_inter" + str(args.interaction_strength)

    print(suffix)

    # NOTE: We first generate all sequences with same length as length_test
    # and then later cut them to required length. Otherwise normalization is
    # messed up (for absolute phase variable).
    print("Generating {} training simulations".format(args.num_train))
    data_train, edges_train = generate_dataset(
        args.num_train, args.length_test, args.sample_freq
    )

    print("Generating {} validation simulations".format(args.num_valid))
    data_valid, edges_valid = generate_dataset(
        args.num_valid, args.length_test, args.sample_freq
    )

    num_timesteps_train = int((args.length / float(args.sample_freq)) - 1)
    data_train = data_train[:, :, :num_timesteps_train, :]
    data_valid = data_valid[:, :, :num_timesteps_train, :]

    print("Generating {} test simulations".format(args.num_test))
    data_test, edges_test = generate_dataset(
        args.num_test, args.length_test, args.sample_freq
    )

    savepath = os.path.expanduser(args.save_dir)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    print("Saving to {}".format(savepath))
    np.save(
        os.path.join(savepath, "feat_train" + suffix + ".npy"),
        data_train,
    )
    np.save(
        os.path.join(savepath, "edges_train" + suffix + ".npy"),
        edges_train,
    )

    np.save(
        os.path.join(savepath, "feat_valid" + suffix + ".npy"),
        data_valid,
    )
    np.save(
        os.path.join(savepath, "edges_valid" + suffix + ".npy"),
        edges_valid,
    )

    np.save(
        os.path.join(savepath, "feat_test" + suffix + ".npy"),
        data_test,
    )
    np.save(
        os.path.join(savepath, "edges_test" + suffix + ".npy"),
        edges_test,
    )