import numpy as np
import networkx as nx
import os
import argparse
from generate_scale_free_n_small_world import save_graph

def adj_to_und(adj, low=True):
    if low:
        adj = (
                np.tril(adj) + np.tril(adj, -1).T
        )
    else:
        adj = (
                np.triu(adj) + np.triu(adj, -1).T
        )
    return adj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network-type', type=str, default='brain_networks',
                        help='What is the type of ground truth edge. (brain_networks, '
                             'chemical_reaction_networks_in_atmosphere, food_webs, gene_coexpression_networks,'
                             'gene_regulatory_networks, intercellular_networks, landscape_networks,'
                             'man-made_organic_reaction_networks, reaction_networks_inside_living_organism,'
                             'social_networks, or vascular_networks)')
    args = parser.parse_args()

    curr_path = os.path.dirname(os.path.realpath(__file__))
    print(curr_path)
    to_grab_path = curr_path + '/' + args.network_type + '/ready/'
    to_save_path = curr_path + '/' + args.network_type + '/un_ready/'

    directed_list = []

    for file in os.listdir(to_grab_path):
        if file.endswith('.npy'):
            # print(os.path.join(edges_dir_path, file))
            directed_list.append(os.path.join(to_grab_path, file))

    for file in directed_list:
        print("Working on :", file)
        adj = np.load(file)
        args.n_balls = adj.shape[0]
        g_old = nx.from_numpy_array(adj, create_using=nx.DiGraph())

        # new_adj = adj_to_und(adj)
        g = g_old.to_undirected()
        # g = nx.from_numpy_matrix(new_adj, create_using=nx.Graph())
        if not nx.is_connected(g):
            g = nx.from_numpy_array(adj_to_und(adj, False), create_using=nx.Graph())
            if not nx.is_connected(g):
                print("-" * 25)
                print("Please replace the adj at: ", file)
                break
        save_name = to_save_path + file.split('/')[-1].split('_')[0]
        # print(save_name)

        save_graph(g, save_name)
