import numpy as np
import torch
import random
import scipy.io
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, jaccard_score
from torch.autograd import Variable


def my_softmax(inputs, axis=1):
    trans_input = inputs.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)


def my_softmax_ssl(input, posi_sets, nega_sets, only_positive, axis=1):
    if not only_positive:
        for index in nega_sets:
            for i in range(len(input)):
                input[i][index] = torch.FloatTensor([1, 0])
    for index in posi_sets:
        for i in range(len(input)):
            input[i][index] = torch.FloatTensor([0, 1])
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)


def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = Variable(y_hard.data - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def one_hot(relation, types=2):
    o_h = torch.nn.functional.one_hot(relation.long(), num_classes=types)
    return o_h


def binary_concrete_sample(logits, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits + Variable(logistic_noise)
    return F.sigmoid(y / tau)


def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)


def top_k_edges(prob_x, k, deconv=False, maskout=False, c_r=False):
    """
    Select the top-k edges from the normalized node vectors
    :param deconv: whether the features come from network deconvolution
    :param prob_x: normalized results by "softmax"
    :param k: number of ture edges from ground truth
    :param maskout: let the function output the masked out top k edge features instead of binary output
    :return: the flattened binary adjacency matrix representing the top-k edges and the masked features
    """
    # check the feasibility of topk:
    num_non_zeros = torch.count_nonzero(prob_x)
    if deconv:
        topk = torch.topk(prob_x, k, dim=-1)
        new_ad = torch.zeros_like(prob_x)
        prob_x_ = prob_x.detach()
    elif c_r:
        topk = torch.topk(prob_x, k, dim=-1)
        new_ad = torch.zeros_like(prob_x)
        prob_x_ = prob_x.detach()
        for element in topk.indices:
            new_ad[element] = 1
        for i in range(len(new_ad)):
            if new_ad[i] == 0:
                prob_x_[i] = 0
        return new_ad, prob_x_
    else:
        topk = torch.topk(prob_x[:, :, 1], k, dim=-1)
        new_ad = torch.zeros_like(prob_x[:, :, 1])
        prob_x_ = prob_x[:, :, 1].detach()
    for i, row in enumerate(topk.indices):
        for element in row:
            new_ad[i][element] = 1
    for i in range(len(new_ad)):
        for j in range(len(new_ad[0])):
            if new_ad[i][j] == 0:
                prob_x_[i][j] = 0
    return new_ad, prob_x_


def top_k_edges_new(prob_x, k, deconv=False, maskout=False, c_r=False):
    """
    Select the top-k edges from the normalized node vectors, the version created on Nov. 11, discard the c_r
    :param deconv: whether the features come from network deconvolution
    :param prob_x: normalized results by "softmax"
    :param k: number of ture edges from ground truth
    :param maskout: let the function output the masked out top k edge features instead of binary output
    :return: the flattened binary adjacency matrix representing the top-k edges and the masked features
    """
    # check the feasibility of topk:
    if deconv:
        prob_features = prob_x
    else:
        prob_features = prob_x[:, :, 1]
    # print("prob_features")
    # print(prob_features.size())
    num_non_zeros = torch.count_nonzero(prob_features, dim=-1)
    # print(num_non_zeros)
    keep_original = []
    if len(num_non_zeros) > 0:
        # print(min(num_non_zeros))
        for i, element in enumerate(num_non_zeros):
            if element < k:
                keep_original.append(i)
    if len(keep_original) == 0:
        topk = torch.topk(prob_features, k, dim=-1)
        new_ad = torch.zeros_like(prob_features)
        prob_x_ = prob_features.detach()

        for i, row in enumerate(topk.indices):
            for element in row:
                new_ad[i][element] = 1
        for i in range(len(new_ad)):
            for j in range(len(new_ad[0])):
                if new_ad[i][j] == 0:
                    prob_x_[i][j] = 0
        return new_ad, prob_x_
    else:
        prob_x_ = prob_features.detach()
        new_ad = []
        for i, adj in enumerate(prob_x_):
            if i in keep_original:
                new_ad_cache = adj
                new_ad_cache[new_ad_cache > 0] = 1
                new_ad_cache = new_ad_cache.view(1, new_ad_cache.size()[0])
                new_ad.append(new_ad_cache)
            else:
                topk = torch.topk(adj, k)
                new_ad_cache = torch.zeros_like(adj)
                for j in topk.indices:
                    new_ad_cache[j] = 1
                new_ad_cache = new_ad_cache.view(1, new_ad_cache.size()[0])
                new_ad.append(new_ad_cache)
                for l in range(len(new_ad_cache[0])):
                    if new_ad_cache[0][l] == 0:
                        adj[l] = 0
        new_ad = torch.cat(new_ad, dim=0)
        return new_ad, prob_x_


def top_k_edges_it(prob_x, k, deconv=False, maskout=False, c_r=False):
    """
    Select the top-k edges from the normalized node vectors for "train_genetic_iterative.py"
    :param deconv: whether the features come from network deconvolution
    :param prob_x: normalized results by "softmax"
    :param k: number of ture edges from ground truth
    :param maskout: let the function output the masked out top k edge features instead of binary output
    :return: the flattened binary adjacency matrix representing the top-k edges and the masked features
    """
    topk = torch.topk(prob_x, k, dim=-1)
    new_ad = torch.zeros_like(prob_x)
    prob_x_ = prob_x.detach()
    for i, row in enumerate(topk.indices):
        for element in row:
            new_ad[i][element] = 1
    for i in range(len(new_ad)):
        for j in range(len(new_ad[0])):
            if new_ad[i][j] == 0:
                prob_x_[i][j] = 0
    return new_ad, prob_x_


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def multi_gumbel_softmax_sample(logits, tau=1, eps=1e-10, rounds=3):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw multiple samples from the Gumbel-Softmax distribution and then average them as the result

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = 0
    i = rounds
    while i > 0:
        gumbel_noise += sample_gumbel(logits.size(), eps=eps)
        i -= 1
    gumbel_noise /= rounds
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, multi_sample=False, rounds=3):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    As an alternative, we can also sample multiple rounds.

    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
      multi_sample: whether to sample multiple rounds
      rounds: the count of multiple samples
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    if not multi_sample:
        y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    else:
        y_soft = multi_gumbel_softmax_sample(logits, tau=tau, eps=eps, rounds=rounds)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def log_influence(logits_encoder, unlabel_feat):
    """
    Multiply the prob from two inputs according to the alignment
    :param logits_encoder:
    :param unlabel_feat:
    :return:
    """
    return logits_encoder * unlabel_feat


def gumbel_softmax_semisl(logits_encoder, unlable_feat, tau=1, hard=False,
                          eps=1e-10, multi_sample=False, rounds=3, log_combin=False):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    First sum up the output from encoder and the features of the unlabelled # todo:
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    As an alternative, we can also sample multiple rounds.

    Args:
      logits_encoder: [batch_size, n_edges, n_class] unnormalized log-probs from the encoder
      unlable_feat: [batch_size, n_edges, n_class]
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
      multi_sample: whether to sample multiple rounds
      rounds: the count of multiple samples
      log_combin: use log-likelihood to combine the information from classifier and the output of the encoder
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    if log_combin:
        logits = log_influence(logits_encoder, unlable_feat)
    else:
        logits = logits_encoder + unlable_feat
    if not multi_sample:
        y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    else:
        y_soft = multi_gumbel_softmax_sample(logits, tau=tau, eps=eps, rounds=rounds)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def gumbel_softmax_ssl(logits, posi_sets, nega_sets, only_positive, tau=1, hard=False, eps=1e-10, multi_sample=False, rounds=3, ):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    As an alternative, we can also sample multiple rounds.
    Specially designed for ssl training
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
      multi_sample: whether to sample multiple rounds
      rounds: the count of multiple samples
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    :param only_positive: control flag
    :param nega_sets: prior knowledge about negative sets
    :param posi_sets: prior knowledge about positive sets
    """
    if not multi_sample:
        y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    else:
        y_soft = multi_gumbel_softmax_sample(logits, tau=tau, eps=eps, rounds=rounds)
    if not only_positive:
        for index in nega_sets:
            for i in range(len(y_soft)):
                y_soft[i][index] = torch.FloatTensor([1, 0])
    for index in posi_sets:
        for i in range(len(y_soft)):
            y_soft[i][index] = torch.FloatTensor([0, 1])
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def multi_gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, rounds=1):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample multiple rounds from the Gumbel-Softmax distribution and optionally discretize.
    The multiple samples will be averaged as the representation.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
      rounds: the number of samples from "gumbel_softmax_sample"
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def netsim_edges(connection):
    edges = np.zeros_like(connection)
    for i, row in enumerate(connection):
        for j, element in enumerate(row):
            if element > 0:
                edges[i][j] = 1
    return edges


def netsim_features(raw_features, valid=False):
    n_total_raw = raw_features.shape[0]
    n_time_raw = raw_features.shape[2]
    if not valid:
        n_time_new = n_time_raw - 49
    else:
        n_time_new = n_time_raw - 99
    features_new = list()
    for i in range(n_time_new):
        features_new.append(raw_features[:, :, i: i + 49, :])
    features_new = np.concatenate(features_new)
    np.random.shuffle(features_new)
    return features_new


def binary_accuracy(output, labels):
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_data_phy2(batch_size=1, suffix='', self_loops=True):
    loc_train = np.load('./data/physics_simulations/loc_train_' + suffix + '.npy')
    vel_train = np.load('./data/physics_simulations/vel_train_' + suffix + '.npy')
    edges_train = np.load('./data/physics_simulations/edges_train_' + suffix + '.npy')

    loc_valid = np.load('./data/physics_simulations/loc_valid_' + suffix + '.npy')
    vel_valid = np.load('./data/physics_simulations/vel_valid_' + suffix + '.npy')
    edges_valid = np.load('./data/physics_simulations/edges_valid_' + suffix + '.npy')

    loc_test = np.load('./data/physics_simulations/loc_test_' + suffix + '.npy')
    vel_test = np.load('./data/physics_simulations/vel_test_' + suffix + '.npy')
    edges_test = np.load('./data/physics_simulations/edges_test_' + suffix + '.npy')

    # [num_samples, num_timesteps, num_dims, num_nodes]
    num_nodes = loc_train.shape[3]

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

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_nodes, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.reshape(edges_train, [-1, num_nodes ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, [-1, num_nodes ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, [-1, num_nodes ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    if not self_loops:
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)),
            [num_nodes, num_nodes])
        edges_train = edges_train[:, off_diag_idx]
        edges_valid = edges_valid[:, off_diag_idx]
        edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def portion_data(raw_data, data_portion, time_steps, shuffle):
    if data_portion == 1.0 and time_steps == 49:
        return raw_data
    if shuffle:
        np.random.shuffle(raw_data)
    num_trajs = raw_data.shape[0]
    num_times = raw_data.shape[0]
    return raw_data[:int(num_trajs * data_portion), :int(time_steps), :, :]


def load_data_phy2_benchmark(args):

    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'

    loc_train = np.load(root_str + 'loc_train_' + keep_str)
    vel_train = np.load(root_str + 'vel_train_' + keep_str)
    edges_train = np.load(root_str + 'edges_train_' + keep_str)

    loc_valid = np.load(root_str + 'loc_valid_' + keep_str)
    vel_valid = np.load(root_str + 'vel_valid_' + keep_str)
    edges_valid = np.load(root_str + 'edges_valid_' + keep_str)

    loc_test = np.load(root_str + 'loc_test_' + keep_str)
    vel_test = np.load(root_str + 'vel_test_' + keep_str)
    edges_test = np.load(root_str + 'edges_test_' + keep_str)

    loc_train = portion_data(loc_train, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_train = portion_data(vel_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    loc_valid = portion_data(loc_valid, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_valid = portion_data(vel_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    # [num_samples, num_timesteps, num_dims, num_nodes]
    num_nodes = loc_train.shape[3]

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

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_nodes, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    # edges_train = np.tile(edges_train, (n_train, 1, 1))
    edges_train = np.reshape(edges_train, [-1, num_nodes ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    # edges_valid = np.tile(edges_valid, (n_valid, 1, 1))
    edges_valid = np.reshape(edges_valid, [-1, num_nodes ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    # edges_test = np.tile(edges_test, (n_test, 1, 1))
    edges_test = np.reshape(edges_test, [-1, num_nodes ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    if not args.include_self_loops:
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)),
            [num_nodes, num_nodes])
        edges_train = edges_train[:, off_diag_idx]
        edges_valid = edges_valid[:, off_diag_idx]
        edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=args.batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=args.batch_size)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def load_data_netsims_benchmark(args):

    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'

    bold_train = np.load(root_str + 'bold_train_' + keep_str)
    edges_train = np.load(root_str + 'edges_train_' + keep_str)

    bold_valid = np.load(root_str + 'bold_valid_' + keep_str)
    edges_valid = np.load(root_str + 'edges_valid_' + keep_str)

    bold_test = np.load(root_str + 'bold_test_' + keep_str)
    edges_test = np.load(root_str + 'edges_test_' + keep_str)

    bold_train = portion_data(bold_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    bold_valid = portion_data(bold_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    num_nodes = bold_train.shape[3]

    n_train = bold_train.shape[0]
    n_test = bold_test.shape[0]
    n_valid = bold_valid.shape[0]

    bold_max = bold_train.max()
    bold_min = bold_train.min()

    bold_train = (bold_train - bold_min) * 2 / (bold_max - bold_min) - 1

    bold_valid = (bold_valid - bold_min) * 2 / (bold_max - bold_min) - 1

    bold_test = (bold_test - bold_min) * 2 / (bold_max - bold_min) - 1

    # Reshape to: [num_sims, num_timesteps, num_nodes, num_dims]

    # Reshape to: [num_sims, num_timesteps, num_nodes, num_dims]
    feat_train = np.transpose(bold_train, [0, 3, 1, 2])
    edges_train = np.tile(edges_train, (n_train, 1, 1))
    edges_train = np.reshape(edges_train, [-1, num_nodes ** 2])

    feat_valid = np.transpose(bold_valid, [0, 3, 1, 2])
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))
    edges_valid = np.reshape(edges_valid, [-1, num_nodes ** 2])

    feat_test = np.transpose(bold_test, [0, 3, 1, 2])
    edges_test = np.tile(edges_test, (n_test, 1, 1))
    edges_test = np.reshape(edges_test, [-1, num_nodes ** 2])

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges: discarded
    # off_diag_idx = np.ravel_multi_index(
    #     np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
    #     [num_atoms, num_atoms])
    # edges_train = edges_train[:, off_diag_idx]
    # edges_valid = edges_valid[:, off_diag_idx]
    # edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=args.batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=args.batch_size)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def load_data_kuramoto(batch_size=1, suffix='', self_loops=True):
    feat_train = np.load('./data/physics_simulations/feat_train_' + suffix + '.npy')
    edges_train = np.load('./data/physics_simulations/edges_train_' + suffix + '.npy')

    feat_valid = np.load('./data/physics_simulations/feat_valid_' + suffix + '.npy')
    edges_valid = np.load('./data/physics_simulations/edges_valid_' + suffix + '.npy')

    feat_test = np.load('./data/physics_simulations/feat_test_' + suffix + '.npy')
    edges_test = np.load('./data/physics_simulations/edges_test_' + suffix + '.npy')

    # [num_samples, num_timesteps, num_dims, num_nodes]
    num_nodes = feat_train.shape[1]
    n_train = feat_train.shape[0]
    n_test = feat_test.shape[0]
    n_valid = feat_valid.shape[0]

    edges_train = np.tile(edges_train, (n_train, 1, 1))
    edges_test = np.tile(edges_test, (n_test, 1, 1))
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))


    # Reshape to: [num_sims, num_atoms * num_atoms]
    edges_train = np.reshape(edges_train, [-1, num_nodes ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_nodes ** 2])
    edges_test = np.reshape(edges_test, [-1, num_nodes ** 2])

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
    if not self_loops:
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)),
            [num_nodes, num_nodes])
        edges_train = edges_train[:, off_diag_idx]
        edges_valid = edges_valid[:, off_diag_idx]
        edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def load_data_netsim(batch_size=1, suffix='', self_loops=True):
    mat = scipy.io.loadmat('./data/netsims/sim' + suffix[-1] + '.mat')

    num_nodes = mat['Nnodes'][0][0]
    num_sims = mat['Nsubjects'][0][0]
    num_time = mat['Ntimepoints'][0][0]

    feat_raw = mat['ts']
    feat_raw = feat_raw.reshape(num_sims, num_nodes, num_time, 1)
    np.random.shuffle(feat_raw)
    feat_train = feat_raw[0: int(num_sims * 0.8)]
    feat_test = feat_raw[int(num_sims * 0.8): int(num_sims * 0.9)]
    feat_valid = feat_raw[int(num_sims * 0.9):]

    feat_train = netsim_features(feat_train)
    feat_test = netsim_features(feat_test)
    feat_valid = netsim_features(feat_valid, valid=True)
    n_train = feat_train.shape[0]
    n_test = feat_test.shape[0]
    n_valid = feat_valid.shape[0]

    info_connection = mat['net']
    edges = netsim_edges(info_connection[0])
    edges_train = np.tile(edges, (n_train, 1, 1))
    edges_test = np.tile(edges, (n_test, 1, 1))
    edges_valid = np.tile(edges, (n_valid, 1, 1))

    # [num_samples, num_timesteps, num_dims, num_nodes]

    # Reshape to: [num_sims, num_atoms * num_atoms]
    edges_train = np.reshape(edges_train, [-1, num_nodes ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_nodes ** 2])
    edges_test = np.reshape(edges_test, [-1, num_nodes ** 2])

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
    if not self_loops:
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)),
            [num_nodes, num_nodes])
        edges_train = edges_train[:, off_diag_idx]
        edges_valid = edges_valid[:, off_diag_idx]
        edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def load_data_genetic(batch_size=1, data_type='LI',
                      self_loops=False, distributed_flag=False, norm_flag=False):
    train_traj = np.load('./data/Synthetic-H/sampled_data/' + data_type + '/train.npy')
    # shape:[num_simulations, num_genes, time_steps]

    n_train = train_traj.shape[0]
    train_traj = np.transpose(train_traj, [0, 2, 1])  # change to [num_simulations, timesteps, num_genes]
    train_traj = train_traj[..., np.newaxis]  # shape: [num_sim, timesteps, num_genes, dimension]
    train_traj = np.transpose(train_traj, [0, 1, 3, 2])  # shape: [num_sim, timesteps, dimension, num_genes]
    edges_train = np.load('./data/Synthetic-H/sampled_data/' + data_type + '/edges.npy')
    edges_train = np.tile(edges_train, (n_train, 1, 1))

    valid_traj = np.load('./data/Synthetic-H/sampled_data/' + data_type + '/valid.npy')
    n_valid = valid_traj.shape[0]
    valid_traj = np.transpose(valid_traj, [0, 2, 1])  # change to [num_simulations, timesteps, num_genes]
    valid_traj = valid_traj[..., np.newaxis]  # shape: [num_sim, timesteps, num_genes, dimension]
    valid_traj = np.transpose(valid_traj, [0, 1, 3, 2])
    edges_valid = np.load('./data/Synthetic-H/sampled_data/' + data_type + '/edges.npy')
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))

    test_traj = np.load('./data/Synthetic-H/sampled_data/' + data_type + '/test.npy')
    n_test = test_traj.shape[0]
    test_traj = np.transpose(test_traj, [0, 2, 1])  # change to [num_simulations, timesteps, num_genes]
    test_traj = test_traj[..., np.newaxis]  # shape: [num_sim, timesteps, num_genes, dimension]
    test_traj = np.transpose(test_traj, [0, 1, 3, 2])
    edges_test = np.load('./data/Synthetic-H/sampled_data/' + data_type + '/edges.npy')
    edges_test = np.tile(edges_test, (n_test, 1, 1))

    # [num_sim, timesteps, dimension, num_genes]
    num_nodes = train_traj.shape[3]

    loc_max = train_traj.max()
    loc_min = train_traj.min()

    if norm_flag:
        # Normalize to [-1, 1]
        norm_train = (train_traj - loc_min) * 2 / (loc_max - loc_min) - 1

        norm_valid = (valid_traj - loc_min) * 2 / (loc_max - loc_min) - 1

        norm_test = (test_traj - loc_min) * 2 / (loc_max - loc_min) - 1
    else:
        norm_train = train_traj
        norm_valid = valid_traj
        norm_test = test_traj

    # Reshape to: [num_sims, num_genes, num_timesteps, num_dims]

    # NOTE: added normalization on Jun.29
    # feat_train = np.transpose(train_traj, [0, 3, 1, 2])  # without normalization
    feat_train = np.transpose(norm_train, [0, 3, 1, 2])
    edges_train = np.reshape(edges_train, [-1, num_nodes ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    # feat_valid = np.transpose(valid_traj, [0, 3, 1, 2])  # without normalization
    feat_valid = np.transpose(norm_valid, [0, 3, 1, 2])
    edges_valid = np.reshape(edges_valid, [-1, num_nodes ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    # feat_test = np.transpose(test_traj, [0, 3, 1, 2])  # without normalization
    feat_test = np.transpose(norm_test, [0, 3, 1, 2])
    edges_test = np.reshape(edges_test, [-1, num_nodes ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    if not self_loops:
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)),
            [num_nodes, num_nodes])
        edges_train = edges_train[:, off_diag_idx]
        edges_valid = edges_valid[:, off_diag_idx]
        edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    if distributed_flag:
        train_sampler = DistributedSampler(train_data)
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, sampler=train_sampler)

    else:
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def sampling_from_data(edges, traj, n_known):
    ind_0, ind_1, ind_2 = np.nonzero(edges)  # because the shape of edges_train !!
    sample_ind_train = random.sample(range(len(ind_0)), n_known)
    train_cl = []  # shape: [num_batch, 2 (nodes), time_steps, num_feature_space]
    train_label = []  # shape: [num_labels]
    for num in sample_ind_train:
        cl_int = []
        train_label.append(edges[ind_0[num]][ind_1[num]][ind_2[num]])
        cl_int.append(traj[ind_0[num], :, :, ind_1[num]])  # send node
        cl_int.append(traj[ind_0[num], :, :, ind_2[num]])  # receive node
        train_cl.append(cl_int)
    train_cl = np.array(train_cl)
    train_label = np.array(train_label)
    return train_cl, train_label


def to_2d_idx(idx, num_cols):
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def encode_onehot_change(labels, num_nodes, dim=0):
    """
    Designed for not fully connected graphs
    :param labels: the unflattened adjacency matrix
    :param num_nodes:
    :param dim: 0 for rel_rec, 1 for rel_send
    :return:
    """
    subs = np.ones((num_nodes, num_nodes))
    sub = np.where(subs)[dim]
    classes = set(sub)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, sub)),
                             dtype=np.int32)
    count = 0
    if dim == 0:  # rel_rec
        for i, row in enumerate(labels):
            for j, element in enumerate(row):
                if element == 0:
                    labels_onehot[count][i] = 0.0
                count += 1
    else:  # rel_send
        for i, row in enumerate(labels):
            for j, element in enumerate(row):
                if element == 0:
                    labels_onehot[count][j] = 0.0
                count += 1

    return labels_onehot


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def metric_calculation(features, binary_features, relations, num_edge_types=2):
    """
    :param num_edge_types:
    :param features: the ith element in numpy array
    :param binary_features: the binary matrices for the edge types (when the number of edge types is 2)
    :param relations: the i th element in numpy ndarray
    :return: auroc, auprc and jac index
    """
    features = array_clear_up(features)
    auroc = roc_auc_score(relations, features, average=None)
    auprc = average_precision_score(relations, features)
    if num_edge_types != 2:
        jac = jaccard_score(relations, binary_features, average='micro')
    else:
        jac = jaccard_score(relations, binary_features)
    return auroc, auprc, jac


def get_minimum_distance(data):
    data = data[:, :, :, :2].transpose(1, 2)
    data_norm = (data ** 2).sum(-1, keepdim=True)
    dist = data_norm + \
           data_norm.transpose(2, 3) - \
           2 * torch.matmul(data, data.transpose(2, 3))
    min_dist, _ = dist.min(1)
    return min_dist.view(min_dist.size(0), -1)


def get_buckets(dist, num_buckets):
    dist = dist.cpu().data.numpy()

    min_dist = np.min(dist)
    max_dist = np.max(dist)
    bucket_size = (max_dist - min_dist) / num_buckets
    thresholds = bucket_size * np.arange(num_buckets)

    bucket_idx = []
    for i in range(num_buckets):
        if i < num_buckets - 1:
            idx = np.where(np.all(np.vstack((dist > thresholds[i],
                                             dist <= thresholds[i + 1])), 0))[0]
        else:
            idx = np.where(dist > thresholds[i])[0]
        bucket_idx.append(idx)

    return bucket_idx, thresholds


def get_correct_per_bucket(bucket_idx, pred, target):
    pred = pred.cpu().numpy()[:, 0]
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def get_correct_per_bucket_(bucket_idx, pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def my_MSE(output, target):
    loss = torch.mean((output - target)**2)
    return loss


def kl_categorical(preds, log_prior, num_nodes, eps=1e-16):
    # print("kl")
    kl_div = preds * (torch.log(preds + eps) - log_prior)  # <- always positive value here
    return kl_div.sum() / (num_nodes * preds.size(0))


def kl_categorical_uniform(preds, num_nodes, num_edge_types, add_const=False,
                           eps=1e-16, auto_correct=True):
    # print("kl_uni")
    kl_div = preds * torch.log(preds + eps)
    if auto_correct:  # add auto correction since it is possible that the kl-divergence may be negative LOL
        if kl_div.sum() < 0:
            const = np.log(num_edge_types)
            kl_div += const
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_nodes * preds.size(0))


def log_standard_categorical(p):
    """Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    Parameters
    ----------
    p: one-hot categorical distribution
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)

    return cross_entropy


def log_like_labels(dims, cuda_flag=False):
    y = [1]
    if dims > 1:
        y.append(0)
        dims -= 1
    y = torch.FloatTensor(y).reshape(1, -1)
    if cuda_flag:
        y = y.cuda()
    loglike_y = -log_standard_categorical(y)
    return loglike_y


def entropy_y(log_prob):
    prob = torch.exp(log_prob)
    return -torch.mul(prob, log_prob).sum(1).mean()


def loss_lxy(log_prob, loglik, loglik_y, kl_div, y_entropy):
    prob = torch.exp(log_prob)
    _Lxy = loglik_y - loglik + kl_div
    q_Lxy = torch.sum(prob * _Lxy.view(-1, 1)) / log_prob.size()[0]
    return q_Lxy + y_entropy


def intergroup_kl_categorical_uniform(preds, num_nodes, num_edge_types, posi_sets, nega_sets, add_const=False,
                                      eps=1e-16):
    """
    This function is utilized to calculate the kl-divergence between either negative or positive sets.
    July 5:At this phase we assume that there are only two elements in either positive sets or negative sets
    :param preds:
    :param num_nodes: (disabled at the moment)
    :param num_edge_types:
    :param nega_sets:
    :param posi_sets:
    :param add_const:
    :param eps:
    :return: one kl_divergence result for each group
    """
    # print("kl_uni")
    # July 5, only implemented the single directional KL-loss
    preds_p = [preds[:, i, :] for i in posi_sets]
    preds_n = [preds[:, i, :] for i in nega_sets]

    # first deal with positive sets
    kl_posi = preds_p[0] * torch.log(preds_p[0] / (preds_p[1] + eps))
    if add_const:
        const = np.log(num_edge_types)
        kl_posi += const

    # then deal with negative sets
    kl_nega = preds_n[0] * torch.log(preds_n[0] / (preds_n[1] + eps))
    if add_const:
        const = np.log(num_edge_types)
        kl_nega += const

    return kl_posi.sum() / preds.size(0), kl_nega.sum() / preds.size(0)


def intergroup_MSE(feature, num_nodes, num_edge_types, posi_sets, nega_sets, add_const=False,
                   eps=1e-16):
    """
    This function is utilized to calculate the MSE between the feature vectors either negative or positive sets.
    July 5:At this phase we assume that there are only two elements in either positive sets or negative sets
    :param feature:
    :param num_nodes: (disabled at the moment)
    :param num_edge_types:
    :param nega_sets:
    :param posi_sets:
    :param add_const:
    :param eps:
    :return: one kl_divergence result for each group
    """
    # print("kl_uni")
    # July 8
    feature_p = [feature[:, i, :] for i in posi_sets]
    feature_n = [feature[:, i, :] for i in nega_sets]

    # first deal with positive sets
    mse_p = my_MSE(feature_p[0], feature_p[1])

    # then deal with negative sets
    mse_n = my_MSE(feature_n[0], feature_n[1])
    return mse_p, mse_n


def intragroup_kl_categorical_uniform(preds, num_nodes, num_edge_types, posi_sets, nega_sets, add_const=False,
                                      eps=1e-16):
    """
    This function is utilized to calculate the kl-divergence between negative and positive sets.
    July 5: At this phase we assume that there are only two elements in either positive sets or negative sets
    :param preds:
    :param num_nodes:
    :param num_edge_types:
    :param nega_sets:
    :param posi_sets:
    :param add_const:
    :param eps:
    :return: an averaged result from the calculated kl-divergences
    """
    # print("kl_uni")
    # July 5, only implemented the single directional KL-loss
    preds_p = [preds[:, i, :] for i in posi_sets]
    preds_n = [preds[:, i, :] for i in nega_sets]

    # first deal with one kl from a sample in positive set to a sample in negative set
    kl_1 = preds_p[0] * torch.log(preds_p[0] / (preds_n[1] + eps))
    if add_const:
        const = np.log(num_edge_types)
        kl_1 += const

    # then deal with another kl from a sample in negative set to a sample in positive set
    kl_2 = preds_n[0] * torch.log(preds_n[0] / (preds_p[1] + eps))
    if add_const:
        const = np.log(num_edge_types)
        kl_2 += const

    return (kl_1.sum() / preds.size(0) + kl_2.sum() / preds.size(0)) / 2


def intragroup_MSE(feature, num_nodes, num_edge_types, posi_sets, nega_sets, add_const=False,
                   eps=1e-16):
    """
    This function is utilized to calculate the MSE between the feature vectors of negative and positive sets.
    July 5:At this phase we assume that there are only two elements in either positive sets or negative sets
    :param feature:
    :param num_nodes: (disabled at the moment)
    :param num_edge_types:
    :param nega_sets:
    :param posi_sets:
    :param add_const:
    :param eps:
    :return: one kl_divergence result for each group
    """
    # print("kl_uni")
    # July 8
    feature_p = [feature[:, i, :] for i in posi_sets]
    feature_n = [feature[:, i, :] for i in nega_sets]

    # first deal with positive [0] -> negative [1] set
    mse_p = my_MSE(feature_p[0], feature_n[1])

    # then deal with negative [0] -> positive [1] sets
    mse_n = my_MSE(feature_n[0], feature_p[1])

    return (mse_p + mse_n) / 2


def nll_gaussian(preds, target, variance, add_const=False):

    if type(target) != type(preds):
        preds = preds[0]
        print("merged!")

    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def edge_accuracy_it(preds, target, threshold=0.5, NDtest=False, topk=False, c_r=False):
    """
    Only for the accuracy calculation in "train_genetic_iterative.py"
    :param threshold:
    :param preds:
    :param target:
    :param NDtest:
    :param topk:
    :param c_r:
    :return:
    """
    edgelist = torch.zeros_like(preds.detach().numpy())
    for i in range(preds.size()[0]):
        for j in range(preds.size()[1]):
            if preds[i][j] >= threshold:
                edgelist[i][j] = 1
    # if not NDtest and not topk:
    #     _, preds = preds.max(-1)
    # # print("preds:shape {}".format(preds.size()))
    correct = edgelist.float().data.eq(
              target.float().data.view_as(edgelist)).cpu().sum()

    return np.float(correct) / (target.size()[0] * target.size()[1])


def adj_thresholding(adj, threshold=0.5, keep_origin=False):
    dim_count = adj.ndim
    if dim_count == 3:
        if not keep_origin:
            adj[adj >= threshold] = 1
        adj[adj < threshold] = 0
    else:
        if not keep_origin:
            adj[adj >= threshold] = 1
        adj[adj < threshold] = 0
    return adj


def adj_normalize(adj, num_nodes):
    """
    Symmetric normalization for adjacency matrix (only with symmetric matrix!!)
    :param num_nodes: number of nodes
    :param adj: np array in the shape of [num_nodes * num_nodes]
    :return: Normalized adjacency matrix in the shape of [num_nodes * num_nodes]
    """
    if adj.shape[0] != num_nodes:
        adj = np.reshape(adj, (num_nodes, num_nodes))
    D = np.diag(np.sum(adj, axis=1))
    D_inv_sqrt = np.linalg.pinv(np.sqrt(D))
    return np.reshape(np.dot(D_inv_sqrt, adj).dot(D_inv_sqrt), num_nodes * num_nodes)


def adj_row_normalize(adj, num_nodes):
    """
    Row wise normalization for adjacency matrix
    :param num_nodes: number of nodes
    :param adj: np array in the shape of [num_nodes * num_nodes]
    :return: Row-wise normalized adjacency matrix in the shape of [num_nodes * num_nodes]
    """
    if adj.shape[0] != num_nodes:
        adj = np.reshape(adj, (num_nodes, num_nodes))
    D = np.diag(np.sum(adj, axis=1))
    D_inv = np.linalg.pinv(D)
    return np.reshape(np.dot(D_inv, adj), num_nodes * num_nodes)


def sparse_dropout(x, rate, noise_shape):
    """
    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1./ (1-rate))

    return out


def adj_col_normalize(adj, num_nodes):
    """
    Column-wise normalization for adjacency matrix
    :param num_nodes: number of nodes
    :param adj: numpy array with the shape of [num_nodes * num_nodes]
    :return: Column-wise normalized adjacency matrix in the shape of [num_nodes * num_nodes]
    """
    if adj.shape[0] != num_nodes:
        adj = np.reshape(adj, (num_nodes, num_nodes))
    D = np.diag(np.sum(adj, axis=0))
    D_inv = np.linalg.pinv(D)
    return np.reshape(np.dot(adj, D_inv), num_nodes * num_nodes)


def adj_it_normalize(adj, num_nodes):
    """
    Naive implementation of row-wise normalization
    :param num_nodes: number of nodes
    :param adj: numpy array with the shape of [num_nodes * num_nodes]
    :return: Normalized adjacency matrix in the shape of [num_nodes * num_nodes]
    """
    if adj.shape[0] != num_nodes:
        adj = np.reshape(adj, (num_nodes, num_nodes))
    row_sums = adj.sum(axis=1, keepdims=True)
    new_matrix = adj / row_sums
    return np.reshape(new_matrix, num_nodes * num_nodes)


def edge_accuracy(preds, target, NDtest=False, topk=False, c_r=False):
    # print("preds_:shape {}".format(preds.size()))
    if not NDtest and not topk:
        _, preds = preds.max(-1)
    # print("preds:shape {}".format(preds.size()))
    if c_r:
        correct = preds.float().data.eq(
            target.float().data).cpu().sum()
        return np.float(correct) / (target.size()[0])
    else:
        correct = preds.float().data.eq(
            target.float().data.view_as(preds)).cpu().sum()

    return np.float(correct) / (target.size()[0] * target.size()[1])


def edge_accuracy_group(preds, target):
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / (target.size(0) * target.size(1))


def metric_report(preds, target, variance, log_prior, num_nodes, num_edge_types, eps=1e-16):
    return


def inter_check(logit_, args):
    """
    output the learned structure during the training process
    :param logit_: the logits from encoder
    :return: a n x n array representing the structure of the network
    """
    _, preds = logit_.max(-1)
    preds_np = preds.cpu().numpy()
    out = np.reshape(preds_np, (len(preds_np), args.num_nodes, args.num_nodes))
    return out


def inter_check_(logit_, num_nodes, cuda):
    """
    output the learned structure during the training process
    :param cuda: flag, whether in cuda or not
    :param num_nodes: the number of nodes
    :param logit_: the logits from encoder
    :return: a n x n array representing the structure of the network
    """
    _, preds = logit_.max(-1)
    if cuda:
        out = preds.view(preds.size()[0], num_nodes, num_nodes)
    else:
        preds_np = preds.cpu().numpy()
        out = np.reshape(preds_np, (len(preds_np), num_nodes, num_nodes))
    return out


def array_clear_up(mat):
    """
    NaN, infinity or too large value detection.
    :param mat: numpy array
    :return: numpy array
    """
    mat = np.nan_to_num(mat, nan=0, posinf=1, neginf=0)
    return mat


def metrics_calculation(edges, truth, edge_types=2):
    auroc = roc_auc_score(truth, edges, average=None)
    auprc = average_precision_score(truth, edges)
    if edge_types != 2:
        jac = jaccard_score(truth, edges, average='micro')
    else:
        jac = jaccard_score(truth, edges)
    return auroc, auprc, jac


def metrics_calculation_no_jac(edges, truth, edge_types=2):
    """
    Only for the metrics of continuous relaxation with layer weights
    :param edges:
    :param truth:
    :param edge_types:
    :return:
    """
    auroc = roc_auc_score(truth, edges, average=None)
    auprc = average_precision_score(truth, edges)

    return auroc, auprc


def dirichlet_energy(adj, data, num_nodes, cuda=False):
    """
    Calculate the Dirichlet energy for smoothness
    :param adj: the adjacency matrix after softmax (the same as used for KL-d) [batch_size, num_nodes ** 2, 2]
    :param data: the input node information [batch_size, num_nodes, time_steps, num_features]
    :param num_nodes: the number of the nodes in the network
    :return: the calculated batch-wise dirichlet energy
    """
    adj_ = adj[:, :, 1].view(adj.size()[0], num_nodes, num_nodes)
    node_features = data.view(data.size()[0], num_nodes, -1)
    d_e = torch.zeros(1)
    if cuda:
        d_e = d_e.cuda()
    for i in range(num_nodes):
        for j in range(num_nodes):
            d_e += (torch.mul(adj_[:, i, j], torch.pow(
                torch.linalg.norm(node_features[:, i, :] - node_features[:, j, :], dim=-1), 2))
                    ).sum()
            # print("i: {}, j:{}".format(i, j))
            # print(d_e)

    return d_e / (num_nodes * num_nodes * adj.size()[0])


def degree_loss(adj, num_nodes, cuda=False):
    """
    The term for degree regularization in iterative training
    :param adj: the adjacency matrix after softmax (the same as used for KL-d) [batch_size, num_nodes ** 2, 2]
    :param num_nodes:  the number of the nodes in the network
    :param cuda:
    :return:
    """
    adj_ = adj[:, :, 1].view(adj.size()[0], num_nodes, num_nodes)
    one_vec = torch.ones(num_nodes)
    front_ones = one_vec.unsqueeze(0)
    back_ones = one_vec.unsqueeze(-1)
    front = front_ones.view(1, front_ones.size()[0], front_ones.size()[1]).repeat(adj.size()[0], 1, 1)
    back = back_ones.view(1, back_ones.size()[0], back_ones.size()[1]).repeat(adj.size()[0], 1, 1)

    if cuda:
        front = front.cuda()
        back = back.cuda()
    return torch.sum(torch.matmul(front, torch.log(torch.matmul(adj_, back) + 1e-12)).squeeze()) / (
            adj_.size()[0] * num_nodes)


def sparsity_loss(adj, num_nodes):
    """
    For the calculation of the F-norm of the Adjacency matrix
    :param adj: the adjacency matrix after softmax (the same as used for KL-d) [batch_size, num_nodes ** 2, 2]
    :param num_nodes: Number of nodes in the network
    :return: The batch-averaged sparsity loss
    """
    adj_ = adj[:, :, 1].view(adj.size()[0], num_nodes, num_nodes)
    return torch.sum(torch.pow(torch.linalg.norm(adj_, ord='fro', dim=(1, 2)), 2)) / (
            adj.size()[0] * num_nodes * num_nodes)


def iip_stop_condition(prev_adj, new_adj, num_nodes=7, th=1e-5, iip_count=3):
    """
    Stop condition calculation for iip-iteration
    :param num_nodes:
    :param prev_adj: flattened adjacency matrix. torch.Tensor. [num_nodes ** 2]
    :param new_adj: flattened adjacency matrix. torch.Tensor. [num_nodes ** 2]
    :param th:
    :param iip_count:
    :return:
    """
    if iip_count <= 2:
        return False
    inter = new_adj - prev_adj
    inter = inter.view(num_nodes, num_nodes)
    fro_norm_cal = torch.pow(torch.linalg.norm(inter, ord='fro'), 2)
    initial_for_norm = th * torch.pow(torch.linalg.norm(torch.ones(num_nodes, num_nodes), ord='fro'), 2)
    return fro_norm_cal < initial_for_norm


def modify_adja_it(matrix, num_random, num_nodes):
    # r_index1 = np.random.randint(0, num_nodes * 2, num_random)
    # r_index2 = np.random.randint(0, num_nodes, num_random)
    # for i in range(num_random):
    #     matrix[r_index1[i]][r_index2[i]] = 1
    r_index1 = np.random.randint(0, num_nodes, num_random)
    for i in range(num_random):
        matrix[r_index1[i]] = 1
    return matrix


def initialization(args):
    # Generate off-diagonal interaction graph
    if not args.include_self_loops:
        adj = np.ones([args.num_nodes, args.num_nodes]) - np.eye(args.num_nodes)

    else:
        adj = np.ones([args.num_nodes, args.num_nodes])

    rel_rec = np.array(encode_onehot(np.where(adj)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(adj)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)  # add "requires_grad = True" on Apr. 30,   NOOOOO!!!! requires_grad=True
    rel_send = torch.FloatTensor(rel_send)
    adj = torch.FloatTensor(adj)
    return adj, rel_rec, rel_send


def load_data(args):
    if args.suffix in ['LI', 'LL', 'CY', 'BF', 'TF', 'BF-CV']:
        train_loader, valid_loader, test_loader = load_data_genetic(
            batch_size=args.batch_size,
            data_type=args.suffix,
            self_loops=args.include_self_loops,
        )
    elif args.suffix in ['springs', 'charged']:
        train_loader, valid_loader, test_loader = load_data_phy2(
            batch_size=args.batch_size,
            suffix=args.suffix,
            self_loops=args.include_self_loops,
        )
    elif args.suffix in ['kuramoto']:
        train_loader, valid_loader, test_loader = load_data_kuramoto(
            batch_size=args.batch_size,
            suffix=args.suffix,
            self_loops=args.include_self_loops,
        )
    else:
        train_loader, valid_loader, test_loader = load_data_netsim(
            batch_size=args.batch_size,
            suffix=args.suffix,
            self_loops=args.include_self_loops,
        )
    return train_loader, valid_loader, test_loader


def load_data_benchmark(args):
    if args.suffix in ['springs']:
        train_loader, valid_loader, test_loader = load_data_phy2_benchmark(
            args=args
        )
    else:
        train_loader, valid_loader, test_loader = load_data_netsims_benchmark(
            args=args
        )
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    print("This is utils.py")
