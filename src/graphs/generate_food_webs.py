import numpy as np
import networkx as nx
import json
import matplotlib.pyplot as plt
import random
from generate_scale_free_n_small_world import property_check
from generate_chemical_reactions_in_atmosphere import save_graph


def plot_graph(nx_graph):
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
    plt.clf()
    # plt.savefig(name_plot)


def threshold_degrees(degree_seq, num_nodes):
    degree_seq = degree_seq.astype(int)
    for i, degree in enumerate(degree_seq):
        if degree > num_nodes:
            degree_seq[i] = num_nodes
        if degree == 0:
            degree_seq[i] = 1
    return degree_seq


def sample_degrees(num_nodes, hyper_scale):
    """
    Follows: https://stackoverflow.com/questions/4328837/generating-a-graph-with-certain-degree-distribution
    :param hyper_scale: denominator of the scale hyperparameter
    :param num_nodes: total number of nodes
    :return: in_degree_seq, out_degree_seq ndarray
    """
    # in_degree_seq = [int(random.expovariate(lambd=num_nodes / 2)) for i in range(num_nodes)]
    in_degree_seq = np.random.exponential(scale=num_nodes / hyper_scale, size=num_nodes)
    # out_degree_seq = [int(random.expovariate(lambd=num_nodes / 2)) for i in range(num_nodes)]
    # out_degree_seq = np.random.exponential(scale=num_nodes / 2, size=num_nodes)
    in_degree_seq = threshold_degrees(in_degree_seq, num_nodes)
    out_degree_seq = in_degree_seq.copy()
    np.random.shuffle(out_degree_seq)
    # out_degree_seq = threshold_degrees(out_degree_seq, num_nodes)
    return in_degree_seq, out_degree_seq


def generate_directed_configuration_networks(
        json_rec, name_networks, hs_space, repetitions
):
    for num_nodes in json_rec['num_nodes']:
        rep = 1
        for hs_v in hs_space:
            for i in range(repetitions):
                if rep >= 15:
                    break
                in_degrees, out_degrees = sample_degrees(num_nodes=num_nodes, hyper_scale=hs_v)
                g = nx.directed_configuration_model(
                    in_degree_sequence=in_degrees,
                    out_degree_sequence=out_degrees
                )
                if property_check(g, json_rec):
                    print("Found rep ", rep)
                    print("Hyper-scale: ", hs_v)
                    file_name = './' + name_networks + '/n' + str(num_nodes) + 'r' + str(rep)
                    print("Save at: " + file_name)
                    save_graph(g, file_name)
                    rep += 1


if __name__ == "__main__":
    total_rep = 150

    search_intervals = 45
    hs_min = 1
    hs_max = 10

    hs = np.linspace(hs_min, hs_max, search_intervals)

    name_space = 'food_webs'
    with open('networks_json_data.json') as json_file:
        json_record = json.load(json_file)
    print(json_record)
    print('-' * 15)
    spec_json_record = json_record[name_space]
    print(spec_json_record)

    generate_directed_configuration_networks(
        json_rec=spec_json_record,
        name_networks=name_space,
        hs_space=hs,
        repetitions=total_rep
    )

    # in_degrees, out_degrees = sample_degrees(num_nodes=20, hyper_scale=)
    # print("in_degrees: ", in_degrees)
    # print("out_degrees: ", out_degrees)
    #
    # g = nx.directed_configuration_model(
    #     in_degree_sequence=in_degrees,
    #     out_degree_sequence=out_degrees
    # )
    # print("Average path length: ", nx.average_shortest_path_length(g))
    # plot_graph(g)
    # # node_degrees = np.random.exponential(scale=10, size=20)
    # # # node_degrees = np.random.Generator.exponential(scale=10.0, size=20)
    # # print(node_degrees)
    # # fig = plt.figure(figsize=(6, 6))

