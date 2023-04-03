import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# from scipy.stats import powerlaw
import powerlaw
import json
from generate_scale_free_n_small_world import get_adjacency, property_check
from generate_man_made_organic_reaction_networks import save_graph, check_power_directed_graph


def generate_directed_scale_free_graphs(
        json_rec, name_networks, alpha_space, beta_space, delta_in_space, delta_out_space,
        in_power, out_power, repetitions
):
    for num_nodes in json_rec['num_nodes']:
        rep = 1
        print("Working on graphs with ", num_nodes, " nodes.")
        for alpha_v in alpha_space:
            beta_space = np.linspace(0.01, 1 - alpha_v - 0.01, search_intervals)
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
                            if property_check(g, json_rec):
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
    total_rep = 150

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

    name_space = "reaction_networks_inside_living_organism"
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
        repetitions=total_rep
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