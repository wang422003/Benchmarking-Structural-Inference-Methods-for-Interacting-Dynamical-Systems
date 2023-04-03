import torch
import config as cfg
from instructors.XNRI import XNRIIns
from instructors.XNRI_enc import XNRIENCIns
from instructors.XNRI_dec import XNRIDECIns
from argparse import ArgumentParser
from utils.general import read_pickle
from models.encoder import AttENC, RNNENC, GNNENC
from models.decoder import GNNDEC, RNNDEC, AttDEC
from models.nri import NRIModel
from torch.nn.parallel import DataParallel
from generate.load import load_kuramoto, load_nri, load_netsims, load_nri_benchmark, load_netsims_benchmark
import numpy as np
import os
import datetime
import time


def init_args():
    parser = ArgumentParser()
    parser.add_argument('--dyn', type=str, default='',
    help='Type of dynamics: springs, charged, kuramoto or netsims.')
    parser.add_argument('--size', type=int, default=5, 
    help='Number of particles.')
    parser.add_argument('--dim', type=int, default=4, 
    help='Dimension of the input states.')
    parser.add_argument('--epochs', type=int, default=500, 
    help='Number of training epochs. 0 for testing.')
    parser.add_argument('--reg', type=float, default=0, 
    help='Penalty factor for the symmetric prior.')
    parser.add_argument('--batch', type=int, default=2 ** 6, help='Batch size.')
    parser.add_argument('--skip', action='store_true', default=True,
    help='Skip the last type of edge.')
    parser.add_argument('--no_reg', action='store_true', default=False,
    help='Omit the regularization term when using the loss as an validation metric.')
    parser.add_argument('--sym', action='store_true', default=False,
    help='Hard symmetric constraint.')
    parser.add_argument('--reduce', type=str, default='cnn',
    help='Method for relation embedding, mlp or cnn.')
    parser.add_argument('--enc', type=str, default='RNNENC', help='Encoder.')
    parser.add_argument('--dec', type=str, default='RNNDEC', help='Decoder.')
    parser.add_argument('--scheme', type=str, default='both',
    help='Training schemes: both, enc or dec.')
    parser.add_argument('--load_path', type=str, default='',
    help='Where to load a pre-trained model.')
    parser.add_argument('--save_folder', type=str, default='logs',
                        help='Where to save the trained model, leave empty to not save anything.')

    # for benchmark:
    parser.add_argument('--data_path', type=str, default='',
    help='Where to load the data. May input the paths to edges_train of the data.')
    parser.add_argument('--save-probs', action='store_true', default=False,
                        help='Save the probs during test.')
    parser.add_argument('--b-network-type', type=str, default='',
                        help='What is the network type of the graph.')
    parser.add_argument('--b-directed', action='store_true', default=False,
                        help='Default choose trajectories from undirected graphs.')
    parser.add_argument('--b-simulation-type', type=str, default='',
                        help='Either springs or netsims.')
    parser.add_argument('--b-suffix', type=str, default='',
                        help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1.'
                             ' Or "50r1" for 50 nodes, rep 1 and noise free.')
    parser.add_argument('--b-portion', type=float, default=1.0,
                        help='Portion of data to be used in benchmarking.')
    parser.add_argument('--b-time-steps', type=int, default=49,
                        help='Portion of time series in data to be used in benchmarking.')
    parser.add_argument('--b-shuffle', action='store_true', default=False,
                        help='Shuffle the data for benchmarking?.')
    parser.add_argument('--b-manual-nodes', type=int, default=0,
                        help='The number of nodes if changed from the original dataset.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    # remember to disable this for submission
    parser.add_argument('--b-walltime', action='store_true', default=True,
                        help='Set wll time for benchmark training and testing. (Max time = 2 days)')
    args = parser.parse_args()
    if args.dyn == '':
        args.dyn = args.b_simulation_type

    if args.data_path == "" and args.b_network_type != "":
        if args.b_directed:
            dir_str = 'directed'
        else:
            dir_str = 'undirected'
        args.data_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/simulations/' + args.b_network_type + '/' + \
                         dir_str + \
                         '/' + args.b_simulation_type + '/edges_train_' + args.b_simulation_type + args.b_suffix + '.npy'
        args.b_manual_nodes = int(args.b_suffix.split('r')[0])
    if args.data_path != '':
        args.size = args.b_manual_nodes

    if args.b_simulation_type == 'springs':
        args.dim = 4
    elif args.b_simulation_type == 'netsims':
        args.dim = 1

    if args.b_time_steps < 49:
        args.reduce = 'mlp'

    return args


def load_data(args):
    path = 'data/{}/{}.pkl'.format(args.dyn, args.size)
    train, val, test = read_pickle(path)
    data = {'train': train, 'val': val, 'test': test}
    return data


def portion_data(raw_data, data_portion, time_steps, shuffle):
    if data_portion == 1.0 and time_steps == 49:
        return raw_data
    if shuffle:
        np.random.shuffle(raw_data)
    num_trajs = raw_data.shape[0]
    num_times = raw_data.shape[0]
    return raw_data[:int(num_trajs * data_portion), :int(time_steps), :, :]


def load_customized_data(args):
    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'
    if args.dyn == 'springs':
        train, val, test = load_customized_springs_data(args, keep_str, root_str)
    elif args.dyn == 'netsims':
        train, val, test = load_customized_netsims_data(args, keep_str, root_str)
    else:
        raise ValueError("Check args.dyn!")
    data = {'train': train, 'val': val, 'test': test}
    return data


def load_customized_springs_data(args, keep_str, root_str):
    loc_train = np.load(root_str + 'loc_train_' + keep_str)
    vel_train = np.load(root_str + 'vel_train_' + keep_str)
    edges_train = np.load(root_str + 'edges_train_' + keep_str)
    edges_train[edges_train > 0] = 1

    loc_valid = np.load(root_str + 'loc_valid_' + keep_str)
    vel_valid = np.load(root_str + 'vel_valid_' + keep_str)
    edges_valid = np.load(root_str + 'edges_valid_' + keep_str)
    edges_valid[edges_valid > 0] = 1

    loc_test = np.load(root_str + 'loc_test_' + keep_str)
    vel_test = np.load(root_str + 'vel_test_' + keep_str)
    edges_test = np.load(root_str + 'edges_test_' + keep_str)
    edges_test[edges_test > 0] = 1

    loc_train = portion_data(loc_train, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_train = portion_data(vel_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    loc_valid = portion_data(loc_valid, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_valid = portion_data(vel_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    loc_test = portion_data(loc_test, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_test = portion_data(vel_test, args.b_portion, args.b_time_steps, args.b_shuffle)

    num_nodes = loc_train.shape[3]

    n_train = loc_train.shape[0]
    n_test = loc_test.shape[0]
    n_valid = loc_valid.shape[0]

    # Reshape to: [num_sims, num_timesteps, num_nodes, num_dims]
    loc_train = np.transpose(loc_train, [0, 1, 3, 2])
    vel_train = np.transpose(vel_train, [0, 1, 3, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.tile(edges_train, (n_train, 1, 1))

    loc_valid = np.transpose(loc_valid, [0, 1, 3, 2])
    vel_valid = np.transpose(vel_valid, [0, 1, 3, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))

    loc_test = np.transpose(loc_test, [0, 1, 3, 2])
    vel_test = np.transpose(vel_test, [0, 1, 3, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.tile(edges_test, (n_test, 1, 1))

    train = list()
    val = list()
    test = list()

    for i in range(n_train):
        train.append((edges_train[i], feat_train[i]))
    for i in range(n_valid):
        val.append((edges_valid[i], feat_valid[i]))
    for i in range(n_test):
        test.append((edges_test[i], feat_test[i]))
    return train, val, test


def load_customized_netsims_data(args, keep_str, root_str):
    bold_train = np.load(root_str + 'bold_train_' + keep_str)
    edges_train = np.load(root_str + 'edges_train_' + keep_str)
    edges_train[edges_train > 0] = 1

    bold_valid = np.load(root_str + 'bold_valid_' + keep_str)
    edges_valid = np.load(root_str + 'edges_valid_' + keep_str)
    edges_valid[edges_valid > 0] = 1

    bold_test = np.load(root_str + 'bold_test_' + keep_str)
    edges_test = np.load(root_str + 'edges_test_' + keep_str)
    edges_test[edges_test > 0] = 1

    bold_train = portion_data(bold_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    bold_valid = portion_data(bold_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    bold_test = portion_data(bold_test, args.b_portion, args.b_time_steps, args.b_shuffle)

    num_nodes = bold_train.shape[3]

    n_train = bold_train.shape[0]
    n_test = bold_test.shape[0]
    n_valid = bold_valid.shape[0]

    # Reshape to: [num_sims, num_timesteps, num_nodes, num_dims]
    feat_train = np.transpose(bold_train, [0, 1, 3, 2])
    edges_train = np.tile(edges_train, (n_train, 1, 1))

    feat_valid = np.transpose(bold_valid, [0, 1, 3, 2])
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))

    feat_test = np.transpose(bold_test, [0, 1, 3, 2])
    edges_test = np.tile(edges_test, (n_test, 1, 1))

    train = list()
    val = list()
    test = list()

    for i in range(n_train):
        train.append((edges_train[i], feat_train[i]))
    for i in range(n_valid):
        val.append((edges_valid[i], feat_valid[i]))
    for i in range(n_test):
        test.append((edges_test[i], feat_test[i]))
    return train, val, test


def run():
    args = init_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cfg.gpu:
        torch.cuda.manual_seed(args.seed)

    # load data
    # original:
    # data = load_data(args)
    start_time = time.time()
    # customized pipeline for saving:
    name_str = args.data_path.split('/')[-3] + '_' + args.data_path.split('/')[-1].split('_', 2)[-1].split('.')[0]
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    save_folder = './{}/MPM-{}-E{}-D{}-exp{}/'.format(args.save_folder, name_str, args.enc,
                                                      args.dec, timestamp)
    save_folder = save_folder.replace(":", "_")
    os.mkdir(save_folder)
    args.save_folder = save_folder
    cfg.init_args(args)
    res_folder = save_folder + 'results/'
    os.mkdir(res_folder)

    # customized:
    data = load_customized_data(args)
    if args.dyn == 'kuramoto':
        data, es, _ = load_kuramoto(data, args.size)

    # original:
    elif args.dyn == 'springs':
        data, es, _ = load_nri(data, args.size)

    # modification for benchmark:
    # elif args.dyn == 'springs':
    #     data, es, _ = load_nri_benchmark(data, args.size, args)

    # original:
    else:
        data, es, _ = load_netsims(data, args.size)
    # modification for benchmark:
    # else:
    #     data, es, _ = load_netsims_benchmark(data, args.size, args)
    if args.data_path != '':
        dim = args.dim if args.reduce == 'cnn' else args.dim * args.b_time_steps
    else:
        dim = args.dim if args.reduce == 'cnn' else args.dim * cfg.train_steps
    encs = {
        'GNNENC': GNNENC,
        'RNNENC': RNNENC,
        'AttENC': AttENC,
    }
    decs = {
        'GNNDEC': GNNDEC,
        'RNNDEC': RNNDEC,
        'AttDEC': AttDEC,
    }
    encoder = encs[args.enc](dim, cfg.n_hid, cfg.edge_type, cfg.do_prob_enc, reducer=args.reduce)
    decoder = decs[args.dec](args.dim, cfg.edge_type, cfg.n_hid, cfg.n_hid, cfg.n_hid, cfg.do_prob_dec,
                             skip_first=args.skip)
    model = NRIModel(encoder, decoder, es, args.size)
    if args.load_path:
        name = 'logs/{}/best.pth'.format(args.load_path)
        model.load_state_dict(torch.load(name)) 
    model = DataParallel(model)
    if cfg.gpu:
        model = model.cuda()
    if args.scheme == 'both':
        # Normal training.
        ins = XNRIIns(model, data, es, args)
    elif args.scheme == 'enc':
        # Only train the encoder.
        ins = XNRIENCIns(model, data, es, args)
    elif args.scheme == 'dec':
        # Only train the decoder.
        ins = XNRIDECIns(model, data, es, args)
    else:
        raise NotImplementedError('training scheme: both, enc or dec')
    ins.train(save_folder, start_time)

    print("Finished.")
    print("Dataset: ", args.dyn)
    print("Ground truth graph locates at: ", args.data_path)
    print("With portion: ", args.b_portion)
    print("With ", args.b_time_steps, " time steps")


if __name__ == "__main__":
    for _ in range(cfg.rounds):
        run()
