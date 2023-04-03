import random
import numpy as np
import networkx as nx
import glob
import os
import matplotlib.pyplot as plt


def get_adjacency(nx_graph):
    return np.array(nx.adjacency_matrix(nx_graph).todense())


def check_attributes(nx_graph):
    """
    Check if the graph contains node attributes
    :param nx_graph:
    :return:
    """
    for node in nx_graph.nodes:
        node_dict = nx_graph.nodes[node]
        if len(node_dict) != 0:
            return True
    return False


def remove_node_attributes(nx_graph):
    """
    Check if the graph contains node attributes
    :param nx_graph:
    :return:
    """
    g_c = nx_graph.copy()
    for node in nx_graph.nodes:
        node_dict = nx_graph.nodes[node]
        if len(node_dict) != 0:
            for key in node_dict:
                del g_c.nodes[node][key]
    return g_c


def save_graph(nx_graph, name_string):
    adj = get_adjacency(nx_graph)
    name_adj = name_string + '_adj.npy'
    name_graphml = name_string + '.graphml'
    name_plot = name_string + '.png'

    in_degree_sequence = sorted((d for n, d in nx_graph.in_degree()), reverse=True)
    out_degree_sequence = sorted((d for n, d in nx_graph.out_degree()), reverse=True)
    dmax_in = max(in_degree_sequence)
    dmax_out = max(out_degree_sequence)

    # save adjacency matrix as .npy
    np.save(name_adj, adj)

    # save the nx graph as .graphml:
    nx.write_graphml(nx_graph, name_graphml)

    # save structure plot, node degree distribution plot
    fig = plt.figure("Degree of the graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    # Gcc = nx_graph.subgraph(sorted(nx.connected_components(nx_graph), key=len, reverse=True)[0])
    # pos = nx.spring_layout(Gcc, seed=10396953)
    # nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    # nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    nx.draw(nx_graph)
    ax0.set_title("Connected components of G")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    # n, bins, patched = ax1.hist(in_degree_sequence)
    ax1.bar(*np.unique(in_degree_sequence, return_counts=True))
    ax1_title = "In-degree histogram"  # with a power " # + str(in_power)
    ax1.set_title(ax1_title)
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("# of Nodes")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(out_degree_sequence, return_counts=True))
    ax2_title = "Out-degree histogram"  # with a power " # + str(out_power)
    ax2.set_title(ax2_title)
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    # plt.show()
    plt.savefig(name_plot)
    plt.clf()



folder_name = 'gene_regulatory_networks'
ratio = 0.2

# locations_adjs = 'home/aoran/Documents/Projects/Benchmark_SI/Local_Project/src/graphs/' + folder_name + '/'

edges_list = []
curr_path = os.path.dirname(os.path.realpath(__file__))

# print(curr_path)
edges_dir_path = curr_path + '/' + folder_name + '/'

for file in os.listdir(edges_dir_path):
    if file.endswith('.npy'):
        # print(os.path.join(edges_dir_path, file))
        edges_list.append(os.path.join(edges_dir_path, file))

for file in edges_list:
    print("Working on: ")
    print(file)
    # Load the adjacency matrix
    edges = np.load(file)
    # print(edges)
    n_balls = edges.shape[0]
    # print(edges.shape)
    seq = list(range(0, n_balls))
    inds = random.sample(seq, int(n_balls * ratio))
    # print(inds)
    for i in inds:
        edges[i][i] = 1
    # print(edges)
    g = nx.from_numpy_matrix(edges, create_using=nx.DiGraph())
    # print(g)
    name_string = file[:-8]
    # print(name_string)
    save_graph(g, name_string)

