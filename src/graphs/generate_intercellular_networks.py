import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from generate_scale_free_n_small_world import get_adjacency, property_check


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


def generate_e_r_graphs(
        json_rec, name_networks, p_space, repetitions
):
    for num_nodes in json_rec['num_nodes']:
        rep = 1
        for p_v in p_space:
            for _ in range(repetitions):
                if rep >= 15:
                    break
                g = nx.erdos_renyi_graph(
                    n=num_nodes,
                    p=p_v,
                    directed=True
                )
                if property_check(g, json_rec):
                    print("Found rep ", rep)
                    print("With prob: ", p_v)
                    file_name = './' + name_networks + '/n' + str(num_nodes) + 'r' + str(rep)
                    print("Save at: " + file_name)
                    save_graph(g, file_name)
                    rep += 1


if __name__ == "__main__":
    total_rep = 15
    # search space:
    search_intervals = 10
    p_prob_min = 0.05
    p_prob_max = 0.75
    p_prob = np.linspace(p_prob_min, p_prob_max, search_intervals)

    name_space = 'intercellular_networks'
    with open('networks_json_data.json') as json_file:
        json_record = json.load(json_file)
    print(json_record)
    print('-' * 15)
    spec_json_record = json_record[name_space]
    print(spec_json_record)
    generate_e_r_graphs(
        json_rec=spec_json_record,
        name_networks=name_space,
        p_space=p_prob,
        repetitions=total_rep
    )

# g = nx.erdos_renyi_graph(
#     n=20,
#     p=0.3,
#     directed=True
# )
# print(len(g.nodes))
# print(len(g.edges))
#
# in_degree_sequence = sorted((d for n, d in g.in_degree()), reverse=True)
# out_degree_sequence = sorted((d for n, d in g.out_degree()), reverse=True)
#
# dmax_in = max(in_degree_sequence)
# dmax_out = max(out_degree_sequence)
#
# print("Average degree: ", )
# print("Density: ", nx.density(g))
# print("Assortativity: ", nx.degree_assortativity_coefficient(g))
# print("Clustering Coefficient: ", nx.average_clustering(g))
# print("Average Shortest Path Length: ", nx.average_shortest_path_length(g))
