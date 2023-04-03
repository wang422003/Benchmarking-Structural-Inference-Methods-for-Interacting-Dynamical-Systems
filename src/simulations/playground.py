import numpy as np
import matplotlib.pyplot as plt
import scipy
import os, glob


# T = 10000
# sample_freq = 10
# n = 10
#
# T_save = int(T / sample_freq - 1)
# bold = np.zeros((T_save, 1, n))
#
# print(bold.shape)
#
# mat = scipy.io.loadmat('sim3.mat')
#
# num_nodes = mat['Nnodes'][0][0]
# num_sims = mat['Nsubjects'][0][0]
# num_time = mat['Ntimepoints'][0][0]
#
# feat_raw = mat['ts']
#
# # print(feat_raw)
# # print(feat_raw.shape)
# # print(feat_raw[0])
# # print("num_sims: ", num_sims)
# feat_new = np.random.randn(100, 1, 15)
# a = np.random.normal(loc=0, scale=0.1, size=feat_new[0].shape)
# print("a: ", a)
# print(a.shape)
#
# c = np.concatenate((np.expand_dims(a, axis=0), feat_new[1:, :, :]), axis=0)
# print(c.shape)
#
# adj = np.random.randn(15, 15)
# res = 0.1 * np.matmul(adj, c[0][0])
# print("res")
#
# print(res)
# print(res.shape)
# c[1][0] = res
# print(c[1][0])
#
# print(c[2])
#
# ext_con = np.random.normal(loc=0.0, scale=1.0, size=15)
# print("ext_con: ", ext_con)
# ext_input = np.random.poisson(lam= 1.0, size=15)
# print("ext_input: ", ext_input)
# input_res = np.multiply(ext_con, ext_input)
# print("input_res: ", input_res)
# print(input_res.shape)
#
# x = np.random.randn(2, 3)
# print(x)
#
# curr_path = os.path.dirname(os.path.realpath(__file__))
# print(curr_path)
# print(curr_path[:-11])
# print(type(curr_path))
#
# curr_path = os.path.dirname(os.path.realpath(__file__))
# edges_dir_path = curr_path[:-11] + 'graphs/' + 'chemical_reaction_networks_in_atmosphere' + '/cache/'

def get_substring_between_two_chars(str_, ch1='r', ch2='_'):
    return str_[::-1][str_[::-1].find(ch2) : str_[::-1].find(ch1)][::-1][:-1]

# for file in os.listdir(edges_dir_path):
#     if file.endswith('.npy'):
#         print(os.path.join(edges_dir_path, file))
#         print(get_substring_between_two_chars(os.path.join(edges_dir_path, file)))

file_name = '/shared/projects/BSIMDS/src/data/Local_Project/src/graphs/chemical_reaction_networks_in_atmosphere/ready/n150r2_adj.npy'
print(get_substring_between_two_chars(file_name))