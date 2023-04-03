"""
This scripy generates simulated social networks.
The original implementation of is to generate directed graphs.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from generate_man_made_organic_reaction_networks import generate_directed_scale_free_graphs


def plot_directed_graph(nx_graph):
    in_degree_sequence = sorted((d for n, d in nx_graph.in_degree()), reverse=True)
    out_degree_sequence = sorted((d for n, d in nx_graph.out_degree()), reverse=True)
    dmax_in = max(in_degree_sequence)
    dmax_out = max(out_degree_sequence)

    # save adjacency matrix as .npy
    # np.save(name_adj, adj)

    # save the nx graph as .graphml:
    # nx.write_graphml(nx_graph, name_graphml)

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
    # dist_in = powerlaw.Fit(in_degree_sequence)
    # print("In-degree alpha: ", dist_in.power_law.alpha)
    ax1.set_title("In-degree histogram")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("# of Nodes")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(out_degree_sequence, return_counts=True))
    # dist_out = powerlaw.Fit(out_degree_sequence)
    # print("Out-degree alpha: ", dist_out.power_law.alpha)
    ax2.set_title("Out-degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.show()
    # plt.savefig(name_plot)


if __name__ == "__main__":
    total_rep = 1500

    search_intervals = 45
    alpha_min = 0.01
    alpha_max = 0.97
    beta_min = 0.01
    beta_max = 0.98
    delta_in_min = 0.01
    delta_in_max = 0.4
    delta_out_min = 0
    delta_out_max = 0.15
    power_in = 2.7
    power_out = 2.1

    alpha = np.linspace(alpha_min, alpha_max, search_intervals)
    beta = np.linspace(beta_min, beta_max, search_intervals)
    delta_in = np.linspace(delta_in_min, delta_in_max, search_intervals)
    delta_out = np.linspace(delta_out_min, delta_out_max, search_intervals)

    name_space = "social_networks"
    with open('networks_json_data.json') as json_file:
        json_record = json.load(json_file)
    print(json_record)
    print('-' * 15)
    spec_json_record = json_record[name_space]
    print(spec_json_record)

    generate_directed_scale_free_graphs(
        json_rec=spec_json_record,
        name_networks=name_space,
        alpha_space=alpha,
        beta_space=beta,
        delta_in_space=delta_in,
        delta_out_space=delta_out,
        in_power=power_in,
        out_power=power_out,
        repetitions=total_rep,
        search_intervals_=search_intervals
    )
    # # g = nx.gnm_random_graph(n=20, p=0.15, directed=True)
    # g = nx.scale_free_graph(n=200)
    # g = nx.DiGraph(g)
    # print("Density: ", nx.density(g))
    # print("Average path length: ", nx.average_shortest_path_length(g))
    # print("Average clustering coefficient: ", nx.average_clustering(g))
    # print("Modularity: ", nx.directed_modularity_matrix(g).sum())
    # plot_directed_graph(g)
    #
    # # continue on this one