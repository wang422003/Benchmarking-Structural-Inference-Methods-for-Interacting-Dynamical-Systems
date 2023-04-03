import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# from scipy.stats import powerlaw
import powerlaw
import json
from generate_scale_free_n_small_world import get_adjacency, property_check


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
    dist_in = powerlaw.Fit(in_degree_sequence)
    print("In-degree alpha: ", dist_in.power_law.alpha)
    ax1.set_title("In-degree histogram")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("# of Nodes")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(out_degree_sequence, return_counts=True))
    dist_out = powerlaw.Fit(out_degree_sequence)
    print("Out-degree alpha: ", dist_out.power_law.alpha)
    ax2.set_title("Out-degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.show()
    # plt.savefig(name_plot)


def save_graph(nx_graph, name_string, in_power, out_power):
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
    ax1_title = "In-degree histogram with a power " + str(in_power)
    ax1.set_title(ax1_title)
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("# of Nodes")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(out_degree_sequence, return_counts=True))
    ax2_title = "Out-degree histogram with a power " + str(out_power)
    ax2.set_title(ax2_title)
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    # plt.show()
    plt.savefig(name_plot)
    plt.clf()


def check_power_directed_graph(nx_g, in_power, out_power):
    in_degree_sequence = sorted((d for n, d in nx_g.in_degree()), reverse=True)
    out_degree_sequence = sorted((d for n, d in nx_g.out_degree()), reverse=True)
    dist_in = powerlaw.Fit(in_degree_sequence)
    dist_out = powerlaw.Fit(out_degree_sequence)
    # print("in_power: ", in_power)
    # print("out_power: ", out_power)
    flag = True
    if (dist_in.power_law.alpha < in_power * 0.75) or (dist_in.power_law.alpha > in_power * 1.25):
        flag = False
    if (dist_out.power_law.alpha < out_power * 0.75) or (dist_out.power_law.alpha > out_power * 1.25):
        flag = False
    return flag, dist_in.power_law.alpha, dist_out.power_law.alpha


def generate_directed_scale_free_graphs(
        json_rec, name_networks, alpha_space, beta_space, delta_in_space, delta_out_space,
        in_power, out_power, repetitions, search_intervals_
):
    for num_nodes in json_rec['num_nodes']:
        rep = 1
        print("Working on graphs with ", num_nodes, " nodes.")
        for alpha_v in alpha_space:
            beta_space = np.linspace(0.01, 1 - alpha_v - 0.01, search_intervals_)
            for beta_v in beta_space:
                for delta_in_v in delta_in_space:
                    for delta_out_v in delta_out_space:
                        gamma = 1 - alpha_v - beta_v
                        for _ in range(repetitions):
                            if rep >= 15:
                                break
                            g = nx.scale_free_graph(
                                n=num_nodes,
                                alpha=alpha_v,
                                beta=beta_v,
                                gamma=gamma,
                                delta_in=delta_in_v,
                                delta_out=delta_out_v
                            )
                            g = nx.DiGraph(g)
                            flag, in_alpha, out_alpha = check_power_directed_graph(g, in_power, out_power)
                            if flag and property_check(g, json_rec):
                                print("Found rep ", rep)
                                print(
                                    "With alpha: ", alpha_v,
                                    ", beta: ", beta_v,
                                    ", gamma: ", gamma,
                                    ", delta_in: ", delta_in_v,
                                    ", delta_out: ", delta_out_v
                                )
                                file_name = './' + name_networks + '/n' + str(num_nodes) + 'r' + str(rep)
                                save_graph(g, file_name, in_alpha, out_alpha)
                                rep += 1


if __name__ == "__main__":
    total_rep = 15

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

    name_space = "man-made_organic_reaction_networks"
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

    # G = nx.scale_free_graph(
    #     n=20,
    #     alpha=0.41,
    #     beta=0.54,
    #     gamma=0.05,
    #     delta_in=0.2,
    #     delta_out=0
    # )
    # G = nx.DiGraph(G)
    # print(len(G.nodes))
    # print(len(G.edges))
    # print("Average degree: ", )
    # print("Clustering Coefficient: ", nx.average_clustering(G))
    # # print("Average Shortest Path Length: ", nx.average_shortest_path_length(G))
    # plot_graph(G)