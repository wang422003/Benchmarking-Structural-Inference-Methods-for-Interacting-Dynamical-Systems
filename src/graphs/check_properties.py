import numpy as np
import networkx as nx
import os
from tqdm import tqdm

# folder_path = './chemical_reaction_networks_in_atmosphere/'
folder_path = './vascular_networks/'
# folder_path = './intercellular_networks/'

file_list = []

clustering_c_max = 0.56
clustering_c_min = 0.45

avg_shortest_path_max = 2.3
avg_shortest_path_min = 1.7

density_max = 0.25
density_min = 0.15

count = 0
for file in os.listdir(folder_path):
    if file.endswith('.graphml'):
        count += 1
        # print(os.path.join(edges_dir_path, file))
        file_list.append(os.path.join(folder_path + file))
        # print(file_list[-1])
        # if count > 150:
        #     break
res_15_ready = []
res_30_ready = []
res_50_ready = []
res_100_ready = []
res_150_ready = []
res_200_ready = []
res_250_ready = []


for file in file_list:
    g = nx.read_graphml(file)
    und_graph = g.to_undirected()
    if not nx.is_connected(und_graph):
        continue
    if not nx.is_strongly_connected(g):
        continue
    nx_c = nx.average_clustering(g)
    nx_s = nx.average_shortest_path_length(g)
    nx_d = nx.density(g)

    # print("Clustering C: ", nx_c)
    # print("Average shortest path length: ", nx_s)
    # print("Density: ", nx_d)

    # if nx_c > clustering_c_max or nx_c < clustering_c_min:
    #     continue
    if nx_s > avg_shortest_path_max or nx_s < avg_shortest_path_min:
        continue
    if nx_d > density_max or nx_d < density_min:
        continue
    print("--Found a graph: ---")
    print(file)
    if file.split('/')[-1][1: 4] == '250':
        res_250_ready.append(file)
    elif file.split('/')[-1][1: 4] == '200':
        res_200_ready.append(file)
    elif file.split('/')[-1][1: 4] == '150':
        res_150_ready.append(file)
    elif file.split('/')[-1][1: 4] == '100':
        res_100_ready.append(file)
    elif file.split('/')[-1][1: 3] == '50':
        res_50_ready.append(file)
    elif file.split('/')[-1][1: 3] == '30':
        res_30_ready.append(file)
    elif file.split('/')[-1][1: 3] == '15':
        res_15_ready.append(file)
    else:
        print("check the name of file.")
    print("Clustering C: ", nx_c)
    print("Average shortest path length: ", nx_s)
    print("Density: ", nx_d)

    print("-" * 25)
print("-----Results-----")
print("15 nodes:")
print(res_15_ready)
print("30 nodes:")
print(res_30_ready)
print("50 nodes:")
print(res_50_ready)
print("100 nodes:")
print(res_100_ready)
print("150 nodes:")
print(res_150_ready)
print("200 nodes:")
print(res_200_ready)
print("250 nodes:")
print(res_250_ready)
    # break