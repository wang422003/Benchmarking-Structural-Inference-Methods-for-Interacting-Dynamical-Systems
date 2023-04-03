import numpy as np
from utils.general import read_pickle
from itertools import permutations
import torch
import pickle
import datetime
import os
import time


def preprocess(data: list, es: np.ndarray):
    """
    Convert the original data to torch.Tensor and organize them in the batch form.

    Args:
        data: [[adj, states], ...], all samples, each contains an adjacency matrix and the node states
        es: edge list

    Return:
        adj: adjacency matrices in the batch form
        states: node states in the batch form
    """
    # data: [[adj, states]]
    adj, state = [np.stack(i, axis=0) for i in zip(*data)]
    # scale the adjacency matrix to {0, 1}, only effective for Charged dataset since the elements take values in {-1, 1}
    adj = (adj + 1) / 2
    row, col = es
    # adjacency matrix in the form of edge list
    adj = adj[:, row, col]
    # organize the data in the batch form
    adj = torch.LongTensor(adj)
    state = torch.FloatTensor(state)
    return adj, state

def load_nri(data: dict, size: int):
    """
    Load Springs / Charged dataset.

    Args:
        data: train / val / test
        size: number of nodes, used for generating the edge list

    Return:
        data: min-max normalized data
        es: edge list
        max_min: maximum and minimum values of each input dimension
    """
    # edge list of a fully-connected graph
    es = np.array(list(permutations(range(size), 2))).T
    # convert the original data to torch.Tensor
    data = {key: preprocess(value, es) for key, value in data.items()}
    # for spring and charged
    # return maximum and minimum values of each input dimension in order to map normalized data back to the original space
    data, max_min = loc_vel(data)
    return data, es, max_min


def load_data():
    path = 'data/{}/{}.pkl'.format('spring', 5)
    train, val, test = read_pickle(path)
    data = {'train': train, 'val': val, 'test': test}
    return data

# path = '/home/aoran/Documents/Projects/Benchmark_SI/Local_Project/src/models/MPM/NRI-MPM-master/data/spring/5.pkl'
# with open(path, 'rb') as f:
#     train, val, test  = pickle.load(f)
# print("Train: ")
# print(len(train))
# print(type(train[0]))
# print(train[0][0].shape)
# print(train[0][1].shape)
#
# print("val: ")
# print(len(val))
# print(val[0][0].shape)
# print(val[0][1].shape)
#
# print("test: ")
# print(len(test))
# print(test[0][0].shape)
# print(test[0][1].shape)


# str = '/home/users/aowang/simulations/gene_coexpression_networks/springs/edges_train_spring15r1_n1.npy'
# str_ls = str.split('/')
# print(str_ls)
# str_last = str_ls[-1].split('_', 2)
# print(str_last)
#
# xx_str = str.split('/')[-1].split('_', 2)[-1]
# print(xx_str)
#
# xxx_str = str.split('/')[-3] + '_' + str.split('/')[-1].split('_', 2)[-1].split('.')[0]
# print("xxx: ", xxx_str)
#
#
# root_str = str[::-1].split('/', 1)[1][::-1]
# print(root_str)


# now = time.time()
# print(now)
# time.sleep(5)
# now_2 = time.time()
# print(now_2 - now)

a = np.random.rand(15, 15)
print(a)

a[a > 0.5] = 1

print(a)

