import os
import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader


def load_data(args):
    loc_max, loc_min, vel_max, vel_min = None, None, None, None
    train_loader, valid_loader, test_loader = None, None, None

    if "kuramoto" in args.suffix:
        train_loader, valid_loader, test_loader = load_ode_data(
            args,
            suffix=args.suffix,
            batch_size=args.batch_size_multiGPU,
            datadir=args.datadir,
        )
    elif "netsim" in args.suffix:
        train_loader, loc_max, loc_min = load_netsim_data(
            batch_size=args.batch_size_multiGPU, datadir=args.datadir
        )
    elif "springs" in args.suffix:
        (
            train_loader,
            valid_loader,
            test_loader,
            loc_max,
            loc_min,
            vel_max,
            vel_min,
        ) = load_springs_data(
            args, args.batch_size_multiGPU, args.suffix, datadir=args.datadir
        )
    elif "LL" or "LI" or "CY" or "BF" or "TF" or "BF-CV" in args.suffix:
        train_loader, valid_loader, test_loader = load_genetic_data(
            args,
            suffix=args.suffix,
            batch_size=args.batch_size_multiGPU,
        )

    else:
        raise NameError("Unknown data to be loaded")

    return train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min


def load_data_customized(args):
    loc_max, loc_min, vel_max, vel_min = None, None, None, None
    train_loader, valid_loader, test_loader = None, None, None

    if "kuramoto" in args.suffix:
        train_loader, valid_loader, test_loader = load_ode_data(
            args,
            suffix=args.suffix,
            batch_size=args.batch_size_multiGPU,
            datadir=args.datadir,
        )
    elif "netsims" in args.suffix:
        (
            train_loader,
            valid_loader,
            test_loader,
            loc_max,
            loc_min,
            vel_max,
            vel_min,
        ) = load_netsims_data_customized(
            args=args,
            batch_size=args.batch_size_multiGPU,
            datadir=args.datadir
        )
    elif "springs" in args.suffix:
        (
            train_loader,
            valid_loader,
            test_loader,
            loc_max,
            loc_min,
            vel_max,
            vel_min,
        ) = load_springs_data_customized(
            args, args.batch_size_multiGPU, args.suffix, datadir=args.datadir
        )
    elif "LL" or "LI" or "CY" or "BF" or "TF" or "BF-CV" in args.suffix:
        train_loader, valid_loader, test_loader = load_genetic_data(
            args,
            suffix=args.suffix,
            batch_size=args.batch_size_multiGPU,
        )

    else:
        raise NameError("Unknown data to be loaded")

    return train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min


def normalize(x, x_min, x_max):
    return (x - x_min) * 2 / (x_max - x_min) - 1


def remove_unobserved_from_data(loc, vel, edge, args):
    loc = loc[:, :, :, : -args.unobserved]
    vel = vel[:, :, :, : -args.unobserved]
    edge = edge[:, : -args.unobserved, : -args.unobserved]
    return loc, vel, edge


def get_off_diag_idx(num_atoms):
    return np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms],
    )

def get_off_diag_idx_customized(num_atoms):
    return np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms))),
        [num_atoms, num_atoms],
    )


def expansion(genetic_loaded_traj):
    genetic_loaded_traj = np.transpose(genetic_loaded_traj, [0, 2, 1])
    genetic_loaded_traj = genetic_loaded_traj[..., np.newaxis]
    genetic_loaded_traj = np.transpose(genetic_loaded_traj, [0, 2, 1, 3])
    return genetic_loaded_traj


def data_preparation(
    loc,
    vel,
    edges,
    loc_min,
    loc_max,
    vel_min,
    vel_max,
    off_diag_idx,
    num_atoms,
    temperature=None,
):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    # Normalize to [-1, 1]
    loc = normalize(loc, loc_min, loc_max)
    vel = normalize(vel, vel_min, vel_max)
    # print("NUm-atoms: ", num_atoms)
    # print("edges: ", edges.shape)

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc = np.transpose(loc, [0, 3, 1, 2])
    vel = np.transpose(vel, [0, 3, 1, 2])
    feat = np.concatenate([loc, vel], axis=3)
    edges = np.reshape(edges, [-1, num_atoms ** 2])
    edges = np.array((edges + 1) / 2, dtype=np.int64)

    feat = torch.FloatTensor(feat)
    edges = torch.LongTensor(edges)

    # print("feat: ", feat.size())
    # print("edges: ", edges.size())
    edges = edges[:, off_diag_idx]

    if temperature is not None:
        dataset = TensorDataset(feat, edges, temperature)
    else:
        dataset = TensorDataset(feat, edges)

    return dataset


def data_preparation_netsims(
    loc,
    edges,
    loc_min,
    loc_max,
    vel_min,
    vel_max,
    off_diag_idx,
    num_atoms,
    temperature=None,
):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    # Normalize to [-1, 1]
    loc = normalize(loc, loc_min, loc_max)

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc = np.transpose(loc, [0, 3, 1, 2])
    feat = loc
    edges = np.reshape(edges, [-1, num_atoms ** 2])
    edges = np.array((edges + 1) / 2, dtype=np.int64)

    feat = torch.FloatTensor(feat)
    edges = torch.LongTensor(edges)

    edges = edges[:, off_diag_idx]

    if temperature is not None:
        dataset = TensorDataset(feat, edges, temperature)
    else:
        dataset = TensorDataset(feat, edges)

    return dataset


def load_springs_data(args, batch_size=1, suffix="", datadir="data"):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    if args.data_path != '':
        print("Loading data from {}".format(datadir))
    else:
        datadir = args.data_path[::-1].split('/', 1)[1][::-1] + '/'
        print("Loading data from {}".format(datadir))

    loc_train = np.load(os.path.join(datadir, "loc_train" + suffix + ".npy"))
    vel_train = np.load(os.path.join(datadir, "vel_train" + suffix + ".npy"))
    edges_train = np.load(os.path.join(datadir, "edges_train" + suffix + ".npy"))

    loc_valid = np.load(os.path.join(datadir, "loc_valid" + suffix + ".npy"))
    vel_valid = np.load(os.path.join(datadir, "vel_valid" + suffix + ".npy"))
    edges_valid = np.load(os.path.join(datadir, "edges_valid" + suffix + ".npy"))

    loc_test = np.load(os.path.join(datadir, "loc_test" + suffix + ".npy"))
    vel_test = np.load(os.path.join(datadir, "vel_test" + suffix + ".npy"))
    edges_test = np.load(os.path.join(datadir, "edges_test" + suffix + ".npy"))

    if args.load_temperatures:
        temperatures_train, temperatures_valid, temperatures_test = load_temperatures(
            suffix=suffix, datadir=datadir
        )
    else:
        temperatures_train, temperatures_valid, temperatures_test = None, None, None

    # [num_samples, num_timesteps, num_dims, num_atoms]
    if args.training_samples != 0:
        loc_train = loc_train[: args.training_samples]
        vel_train = vel_train[: args.training_samples]
        edges_train = edges_train[: args.training_samples]

    if args.test_samples != 0:
        loc_test = loc_test[: args.test_samples]
        vel_test = vel_test[: args.test_samples]
        edges_test = edges_test[: args.test_samples]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Exclude self edges
    off_diag_idx = get_off_diag_idx(args.num_atoms)

    train_data = data_preparation(
        loc_train,
        vel_train,
        edges_train,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_train,
    )
    valid_data = data_preparation(
        loc_valid,
        vel_valid,
        edges_valid,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_valid,
    )
    test_data = data_preparation(
        loc_test,
        vel_test,
        edges_test,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_test,
    )
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8
    )
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    return (
        train_data_loader,
        valid_data_loader,
        test_data_loader,
        loc_max,
        loc_min,
        vel_max,
        vel_min,
    )


def load_springs_data_customized(args, batch_size=1, suffix="", datadir="data"):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    if args.data_path != '':
        print("Loading data from {}".format(args.data_path))
        datadir = args.data_path[::-1].split('/', 1)[1][::-1] + '/'
        suffix = args.data_path[::-1].split('/')[0][::-1].split('_', 2)[-1].split('.')[0]
    else:
        datadir = args.data_path[::-1].split('/', 1)[1][::-1] + '/'
        print("Loading data from {}".format(datadir))

    loc_train = np.load(os.path.join(datadir, "loc_train_" + suffix + ".npy"))
    vel_train = np.load(os.path.join(datadir, "vel_train_" + suffix + ".npy"))
    edges_train = np.load(os.path.join(datadir, "edges_train_" + suffix + ".npy"))
    edges_train[edges_train > 0] = 1

    loc_valid = np.load(os.path.join(datadir, "loc_valid_" + suffix + ".npy"))
    vel_valid = np.load(os.path.join(datadir, "vel_valid_" + suffix + ".npy"))
    edges_valid = np.load(os.path.join(datadir, "edges_valid_" + suffix + ".npy"))
    edges_valid[edges_valid > 0] = 1

    loc_test = np.load(os.path.join(datadir, "loc_test_" + suffix + ".npy"))
    vel_test = np.load(os.path.join(datadir, "vel_test_" + suffix + ".npy"))
    edges_test = np.load(os.path.join(datadir, "edges_test_" + suffix + ".npy"))
    edges_test[edges_test > 0] = 1

    if args.load_temperatures:
        temperatures_train, temperatures_valid, temperatures_test = load_temperatures(
            suffix=suffix, datadir=datadir
        )
    else:
        temperatures_train, temperatures_valid, temperatures_test = None, None, None

    # [num_samples, num_timesteps, num_dims, num_atoms]
    if args.training_samples != 0:
        loc_train = loc_train[: args.training_samples]
        vel_train = vel_train[: args.training_samples]
        edges_train = edges_train[: args.training_samples]

    if args.test_samples != 0:
        loc_test = loc_test[: args.test_samples]
        vel_test = vel_test[: args.test_samples]
        edges_test = edges_test[: args.test_samples]

    loc_train = portion_data(loc_train, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_train = portion_data(vel_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    loc_valid = portion_data(loc_valid, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_valid = portion_data(vel_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    n_train = loc_train.shape[0]
    n_test = loc_test.shape[0]
    n_valid = loc_valid.shape[0]

    edges_train = np.tile(edges_train, (n_train, 1, 1))
    edges_test = np.tile(edges_test, (n_test, 1, 1))
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Exclude self edges
    off_diag_idx = get_off_diag_idx_customized(args.num_atoms)

    train_data = data_preparation(
        loc_train,
        vel_train,
        edges_train,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_train,
    )
    valid_data = data_preparation(
        loc_valid,
        vel_valid,
        edges_valid,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_valid,
    )
    test_data = data_preparation(
        loc_test,
        vel_test,
        edges_test,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_test,
    )
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8
    )
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    return (
        train_data_loader,
        valid_data_loader,
        test_data_loader,
        loc_max,
        loc_min,
        vel_max,
        vel_min,
    )


def portion_data(raw_data, data_portion, time_steps, shuffle):
    if data_portion == 1.0 and time_steps == 49:
        return raw_data
    if shuffle:
        np.random.shuffle(raw_data)
    num_trajs = raw_data.shape[0]
    num_times = raw_data.shape[0]
    return raw_data[:int(num_trajs * data_portion), :int(time_steps), :, :]


def load_netsims_data_customized(args, batch_size=1, suffix="", datadir="data"):
    if args.data_path != '':
        print("Loading data from {}".format(args.data_path))
        datadir = args.data_path[::-1].split('/', 1)[1][::-1] + '/'
        suffix = args.data_path[::-1].split('/')[0][::-1].split('_', 2)[-1].split('.')[0]
    else:
        datadir = args.data_path[::-1].split('/', 1)[1][::-1] + '/'
        print("Loading data from {}".format(datadir))

    bold_train = np.load(datadir + 'bold_train_' + suffix + '.npy')
    edges_train = np.load(datadir + 'edges_train_' + suffix + '.npy')
    edges_train[edges_train > 0] = 1

    bold_valid = np.load(datadir + 'bold_valid_' + suffix + '.npy')
    edges_valid = np.load(datadir + 'edges_valid_' + suffix + '.npy')
    edges_valid[edges_valid > 0] = 1

    bold_test = np.load(datadir + 'bold_test_' + suffix + '.npy')
    edges_test = np.load(datadir + 'edges_test_' + suffix + '.npy')
    edges_test[edges_test > 0] = 1

    bold_train = portion_data(bold_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    bold_valid = portion_data(bold_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    if args.load_temperatures:
        temperatures_train, temperatures_valid, temperatures_test = load_temperatures(
            suffix=suffix, datadir=datadir
        )
    else:
        temperatures_train, temperatures_valid, temperatures_test = None, None, None
    num_nodes = bold_train.shape[3]
    args.num_atoms = num_nodes

    n_train = bold_train.shape[0]
    n_test = bold_test.shape[0]
    n_valid = bold_valid.shape[0]
    edges_train = np.tile(edges_train, (n_train, 1, 1))
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))
    edges_test = np.tile(edges_test, (n_test, 1, 1))

    # [num_samples, num_timesteps, num_dims, num_atoms]
    if args.training_samples != 0:
        bold_train = bold_train[: args.training_samples]
        edges_train = edges_train[: args.training_samples]

    if args.test_samples != 0:
        bold_test = bold_test[: args.test_samples]
        edges_test = edges_test[: args.test_samples]

    loc_max = bold_train.max()
    loc_min = bold_train.min()
    vel_max = bold_train.max()
    vel_min = bold_train.min()

    # Exclude self edges
    off_diag_idx = get_off_diag_idx_customized(args.num_atoms)

    train_data = data_preparation_netsims(
        bold_train,
        edges_train,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_train,
    )
    valid_data = data_preparation_netsims(
        bold_valid,
        edges_valid,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_valid,
    )

    test_data = data_preparation_netsims(
        bold_test,
        edges_test,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_test,
    )
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8
    )
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    return (
        train_data_loader,
        valid_data_loader,
        test_data_loader,
        loc_max,
        loc_min,
        vel_max,
        vel_min,
    )


def load_temperatures(suffix="", datadir="data"):
    temperatures_train = np.load(
        os.path.join(datadir, "temperatures_train" + suffix + ".npy")
    )
    temperatures_valid = np.load(
        os.path.join(datadir, "temperatures_valid" + suffix + ".npy")
    )
    temperatures_test = np.load(
        os.path.join(datadir, "temperatures_test" + suffix + ".npy")
    )

    temperatures_train = torch.FloatTensor(temperatures_train)
    temperatures_valid = torch.FloatTensor(temperatures_valid)
    temperatures_test = torch.FloatTensor(temperatures_test)

    return temperatures_train, temperatures_valid, temperatures_test


def load_ode_data(args, batch_size=1, suffix="", datadir="data"):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    feat_train = np.load(os.path.join(datadir, "feat_train" + suffix + ".npy"))
    edges_train = np.load(os.path.join(datadir, "edges_train" + suffix + ".npy"))
    feat_valid = np.load(os.path.join(datadir, "feat_valid" + suffix + ".npy"))
    edges_valid = np.load(os.path.join(datadir, "edges_valid" + suffix + ".npy"))
    feat_test = np.load(os.path.join(datadir, "feat_test" + suffix + ".npy"))
    edges_test = np.load(os.path.join(datadir, "edges_test" + suffix + ".npy"))

    # [num_sims, num_atoms, num_timesteps, num_dims]
    num_atoms = feat_train.shape[1]
    if args.training_samples != 0:
        feat_train = feat_train[: args.training_samples]
        edges_train = edges_train[: args.training_samples]

    if args.test_samples != 0:
        feat_test = feat_test[: args.test_samples]
        edges_test = edges_test[: args.test_samples]

    # Reshape to: [num_sims, num_atoms * num_atoms]
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    edges_train = edges_train / np.max(edges_train)
    edges_valid = edges_valid / np.max(edges_valid)
    edges_test = edges_test / np.max(edges_test)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = get_off_diag_idx(num_atoms)
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )  # , num_workers=8
    # )
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(
        test_data, batch_size=batch_size
    )  # , num_workers=8) ##THIS

    return train_data_loader, valid_data_loader, test_data_loader


def load_genetic_data(args, batch_size=1, suffix="", datadir='./Synthetic-H/sampled_data/'):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    datadir = datadir + suffix
    feat_train = np.load(os.path.join(datadir, "train.npy"))
    edges_train = np.load(os.path.join(datadir, "edges.npy"))
    feat_valid = np.load(os.path.join(datadir, "valid.npy"))
    edges_valid = np.load(os.path.join(datadir, "edges.npy"))
    feat_test = np.load(os.path.join(datadir, "test.npy"))
    edges_test = np.load(os.path.join(datadir, "edges.npy"))

    feat_train = expansion(feat_train)
    feat_valid = expansion(feat_valid)
    feat_test = expansion(feat_test)
    # print("Loaded: ")
    # print("feat_train: {}".format(feat_train.shape))
    # print("edges_train: {}".format(edges_train.shape))
    # print("feat_valid: {}".format(feat_valid.shape))
    # print("edges_valid: {}".format(edges_valid.shape))
    # print("feat_test: {}".format(feat_test.shape))
    # print("edges_test: {}".format(edges_test.shape))
    # [num_sims, num_atoms, num_timesteps, num_dims]

    num_atoms = feat_train.shape[1]
    n_train = feat_train.shape[0]
    n_valid = feat_valid.shape[0]
    n_test = feat_test.shape[0]
    edges_train = np.tile(edges_train, (n_train, 1, 1))
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))
    edges_test = np.tile(edges_test, (n_test, 1, 1))

    if args.training_samples != 0:
        feat_train = feat_train[: args.training_samples]
        edges_train = edges_train[: args.training_samples]

    if args.test_samples != 0:
        feat_test = feat_test[: args.test_samples]
        edges_test = edges_test[: args.test_samples]

    # Reshape to: [num_sims, num_atoms * num_atoms]
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    edges_train = edges_train / np.max(edges_train)
    edges_valid = edges_valid / np.max(edges_valid)
    edges_test = edges_test / np.max(edges_test)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = get_off_diag_idx(num_atoms)
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )  # , num_workers=8
    # )
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(
        test_data, batch_size=batch_size
    )  # , num_workers=8) ##THIS

    return train_data_loader, valid_data_loader, test_data_loader



def load_data_for_ncg(datadir, data_index, suffix):
    """Data loading for Neural Granger Causality method (one example at a time)."""
    feat_train = np.load(os.path.join(datadir, "feat_train_small" + suffix + ".npy"))
    edges_train = np.load(os.path.join(datadir, "edges_train_small" + suffix + ".npy"))
    return feat_train[data_index], edges_train[data_index]


def load_netsim_data(batch_size=1, datadir="data"):
    print("Loading data from {}".format(datadir))

    subject_id = [1, 2, 3, 4, 5]

    print("Loading data for subjects ", subject_id)

    loc_train = torch.zeros(len(subject_id), 15, 200)
    edges_train = torch.zeros(len(subject_id), 15, 15)

    for idx, elem in enumerate(subject_id):
        fileName = "sim3_subject_%s.npz" % (elem)
        ld = np.load(os.path.join(datadir, "netsim", fileName))
        loc_train[idx] = torch.FloatTensor(ld["X_np"])
        edges_train[idx] = torch.LongTensor(ld["Gref"])

    # [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = loc_train.unsqueeze(-1)

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    loc_train = normalize(loc_train, loc_min, loc_max)

    # Exclude self edges
    num_atoms = loc_train.shape[1]

    off_diag_idx = get_off_diag_idx(num_atoms)
    edges_train = torch.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = (edges_train + 1) // 2
    edges_train = edges_train[:, off_diag_idx]

    train_data = TensorDataset(loc_train, edges_train)

    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8
    )

    return (train_data_loader, loc_max, loc_min)


def unpack_batches(args, minibatch):
    if args.load_temperatures:
        (data, relations, temperatures) = minibatch
    else:
        (data, relations) = minibatch
        temperatures = None
    if args.cuda:
        data, relations = data.cuda(), relations.cuda()
        if args.load_temperatures:
            temperatures = temperatures.cuda()
    return data, relations, temperatures
