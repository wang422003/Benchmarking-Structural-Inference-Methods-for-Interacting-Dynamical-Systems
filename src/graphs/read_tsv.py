"""
Read the produced .tsv edgelists for the generation of universal format of networks
"""
import pandas as pd
import networkx as nx
import numpy as np
from generate_scale_free_n_small_world import get_adjacency
import matplotlib.pyplot as plt
import json


def save_directed_graph(nx_graph, name_string,):
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
    ax1.bar(*np.unique(in_degree_sequence, return_counts=True))
    ax1.set_title("In-degree histogram")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("# of Nodes")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(out_degree_sequence, return_counts=True))
    ax2.set_title("Out-degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.savefig(name_plot)


if __name__ == "__main__":

    path1 = '/home/aoran/Documents/Projects/Benchmark_SI/Local_Project/src/graphs/network_generation_algo/snippets/test_networks/fflatt_transcriptional_network_0_nodes_250_ffl_perc_0.2.tsv'
    path = '/home/aoran/Documents/Projects/Benchmark_SI/Local_Project/src/graphs/network_generation_algo/snippets/test_networks/fflatt_transcriptional_network_0_nodes_28_ffl_perc_0.2.tsv'
    folder_path = '/home/aoran/Documents/Projects/Benchmark_SI/Local_Project/src/graphs/network_generation_algo/snippets/test_networks/cache/'

    name_space = 'gene_regulatory_networks'
    with open('networks_json_data.json') as json_file:
        json_record = json.load(json_file)
    spec_json_record = json_record[name_space]
    for num_nodes in spec_json_record['num_nodes']:
        this_path = folder_path + 'fflatt_transcriptional_network_0_nodes_' + str(num_nodes) + '_ffl_perc_0.2.tsv'
        file_name = './' + name_space + 'n' + str(num_nodes) + 'r' + str(0)

        df = pd.read_csv(this_path, sep='\t', header=None)
        print(df)
        print(df.columns)

        # G = nx.from_pandas_edgelist(df, source=0, target=1)
        G = nx.read_edgelist(this_path, create_using=nx.DiGraph())
        print(len(G.nodes))
        print(len(G.edges))
        print(nx.average_clustering(G))
        print(nx.average_shortest_path_length(G))
        print(np.array(nx.adjacency_matrix(G).todense()))
        print([G.in_degree()])
        print([G.out_degree()])
        print("Save at: " + file_name)
        save_directed_graph(G, file_name)

