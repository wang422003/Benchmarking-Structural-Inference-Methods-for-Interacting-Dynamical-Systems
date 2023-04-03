# The original code was taken from:
# https://github.com/loeweX/AmortizedCausalDiscovery/blob/master/codebase/data/generate_dataset.py
# With modification

from synthetic_sim import ChargedParticlesSim, SpringSim
import time
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='springs',
                    help='What simulation to generate.')
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
parser.add_argument('--n-balls', type=int, default=50,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument("--save-dir", type=str, default="./Data/springs", help="Where to save generated data.")
parser.add_argument('--full-graph', action='store_true', default=False,
                    help='Whether to generate a fully-connected network.')
parser.add_argument('--noise', action='store_true', default=False,
                    help='Whether to add Gaussian noises to the features.')
parser.add_argument('--noise-var', type=float, default=0.2,
                    help='The level of Gaussian noise.')
parser.add_argument('--noise-level', type=int, default=5,
                    help='The level of Gaussian noise.')

args = parser.parse_args()

if args.noise:
    noise_var = args.noise_var
else:
    noise_var = 0.0

if args.simulation == 'springs':
    sim = SpringSim(noise_var=noise_var, n_balls=args.n_balls, full_graph=args.full_graph)
    if args.full_graph:
        suffix = '_springs' + str(args.n_balls) + 'full'
    else:
        suffix = '_springs' + str(args.n_balls)
    # if args.noise:
    #     suffix += '_N' + str(args.noise_level)
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=noise_var, n_balls=args.n_balls)
    suffix = '_charged' + str(args.n_balls)
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

# suffix += str(args.n_balls)
np.random.seed(args.seed)

print(suffix)


def generate_dataset(num_sims, length, sample_freq):
    # concat as [loc, vel]
    loc_all = list()
    vel_all = list()
    features_all = list()
    # edges_all = list()
    edges = np.zeros(1)

    for i in range(num_sims):
        t = time.time()
        loc, vel, edges = sim.sample_trajectory(T=length,
                                                sample_freq=sample_freq)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)
        vel_all.append(vel)
        # edges_all.append(edges)

    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)

    return loc_all, vel_all, edges


def generate_dataset_noise(num_sims, length, sample_freq, noise_level):
    # concat as [loc, vel]
    loc_all = list()
    vel_all = list()
    features_all = list()
    # edges_all = list()
    edges = np.zeros(1)

    for i in range(num_sims):
        t = time.time()
        loc, vel, edges = sim.sample_trajectory_noise_level(
            noise_level=noise_level,
            T=length,
            sample_freq=sample_freq
        )
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        # print("loc len: ", len(loc))
        # print("loc vel: ", len(vel))
        loc_all.append(np.array(loc))
        vel_all.append(np.array(vel))
        # edges_all.append(edges)
    loc_all = np.array(loc_all)
    vel_all = np.array(vel_all)
    # print("loc_all np: ", loc_all.shape)
    # print("vel_all np: ", vel_all.shape)
    loc_all = np.transpose(loc_all, axes=[1, 0, 2, 3, 4])
    vel_all = np.transpose(vel_all, axes=[1, 0, 2, 3, 4])  # (8000, 49, 2, 10)
    # print(stack_vel[0].shape)
    # loc_all = [np.stack(loc_all[jj][ii]) for jj in range(num_sims) for ii in range(noise_level)]
    # vel_all = [np.stack(vel_all[jj][ii]) for jj in range(num_sims) for ii in range(noise_level)]
    # loc_all = [np.stack(loc_all[ii]) for ii in range(noise_level)]
    # vel_all = [np.stack(vel_all[ii]) for ii in range(noise_level)]
    # print("len loc_all for noise: ", len(loc_all))
    # print(loc_all[0].shape)
    # print("len vel_all for noise: ", len(vel_all))
    # print(vel_all[1].shape)
    return loc_all, vel_all, edges


if not args.noise:
    print("-----Generating datasets without noise-----")
    print("Generating {} training simulations".format(args.num_train))
    loc_train, vel_train, edges_train = generate_dataset(args.num_train,
                                                         args.length,
                                                         args.sample_freq)

    print("Generating {} validation simulations".format(args.num_valid))
    loc_valid, vel_valid, edges_valid = generate_dataset(args.num_valid,
                                                         args.length,
                                                         args.sample_freq)

    print("Generating {} test simulations".format(args.num_test))
    loc_test, vel_test, edges_test = generate_dataset(args.num_test,
                                                      args.length_test,
                                                      args.sample_freq)

    savepath = os.path.expanduser(args.save_dir)
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

else:
    print("-----Generating datasets with noise-----")
    print("Generating {} training simulations".format(args.num_train))
    loc_train, vel_train, edges_train = generate_dataset_noise(
        args.num_train,
        args.length,
        args.sample_freq,
        args.noise_level
    )

    print("Generating {} validation simulations".format(args.num_valid))
    loc_valid, vel_valid, edges_valid = generate_dataset_noise(
        args.num_valid,
        args.length,
        args.sample_freq,
        args.noise_level
    )

    print("Generating {} test simulations".format(args.num_test))
    loc_test, vel_test, edges_test = generate_dataset_noise(
        args.num_test,
        args.length_test,
        args.sample_freq,
        args.noise_level
    )
    path_new = args.save_dir + '/noise_data'
    savepath = os.path.expanduser(path_new)

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    np.save(os.path.join(savepath, 'edges_train' + suffix + '.npy'), edges_train)
    np.save(os.path.join(savepath, 'edges_valid' + suffix + '.npy'), edges_valid)
    np.save(os.path.join(savepath, 'edges_test' + suffix + '.npy'), edges_test)

    for j in range(args.noise_level):
        level = j + 1
        np.save(os.path.join(savepath, 'loc_train' + suffix + 'n' + str(level) + '.npy'), loc_train[j])
        np.save(os.path.join(savepath, 'vel_train' + suffix + 'n' + str(level) + '.npy'), vel_train[j])

        np.save(os.path.join(savepath, 'loc_valid' + suffix + 'n' + str(level) + '.npy'), loc_valid[j])
        np.save(os.path.join(savepath, 'vel_valid' + suffix + 'n' + str(level) + '.npy'), vel_valid[j])

        np.save(os.path.join(savepath, 'loc_test' + suffix + 'n' + str(level) + '.npy'), loc_test[j])
        np.save(os.path.join(savepath, 'vel_test' + suffix + 'n' + str(level) + '.npy'), vel_test[j])

