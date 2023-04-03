"""
This script generates the simulated landscape networks.
The networks are undirected.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from generate_scale_free_n_small_world import save_graph


def plot_undirected_graph(nx_graph, name_string='',):
    # adj = get_adjacency(nx_graph)
    # name_adj = name_string + '_adj.npy'
    # name_graphml = name_string + '.graphml'
    # name_plot = name_string + '.png'

    degree_sequence = sorted((d for n, d in nx_graph.degree()), reverse=True)
    dmax = max(degree_sequence)

    # save adjacency matrix as .npy
    # np.save(name_adj, adj)

    # save the nx graph as .graphml:
    # nx.write_graphml(nx_graph, name_graphml)

    # save structure plot, node degree distribution plot
    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = nx_graph.subgraph(sorted(nx.connected_components(nx_graph), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Connected components of G")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.show()
    # plt.savefig(name_plot)


def generate_geographical_networks(json_record, name_networks, theta_space):
    for num_nodes in json_record['num_nodes']:
        rep = 1
        for theta_v in theta_space:
            for i in range(3):
                if rep >= 15:
                    break
                g = nx.geographical_threshold_graph(
                    n=num_nodes,
                    theta=int(num_nodes * theta_v),
                )
                # print("Type of g: ", type(g))
                # print(g)
                file_name = './' + name_networks + '/n' + str(num_nodes) + 'r' + str(rep) + '_theta_' + str(theta_v)
                print("Save at: " + file_name)
                save_graph(g, file_name)
                rep += 1


if __name__ == "__main__":

    theta = np.array([0.5, 1.0, 2.0])

    name_space = 'landscape_networks'
    with open('networks_json_data.json') as json_file:
        json_record = json.load(json_file)

    print(json_record)
    print('-' * 15)
    spec_json_record = json_record[name_space]
    print(spec_json_record)

    generate_geographical_networks(
        json_record=spec_json_record,
        name_networks=name_space,
        theta_space=theta
    )
    #
    # g = nx.geographical_threshold_graph(
    #     n=100,
    #     theta=200,
    # )
    # plot_undirected_graph(g)

