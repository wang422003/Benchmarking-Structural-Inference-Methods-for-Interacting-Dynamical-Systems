import numpy as np
import networkx as nx
from networkx import NetworkXError
from generate_man_made_organic_reaction_networks import plot_graph, save_graph
import powerlaw
import json


def check_power_directed_graph(nx_g, in_power, out_power):
    in_degree_sequence = sorted((d for n, d in nx_g.in_degree()), reverse=True)
    out_degree_sequence = sorted((d for n, d in nx_g.out_degree()), reverse=True)
    dist_in = powerlaw.Fit(in_degree_sequence)
    dist_out = powerlaw.Fit(out_degree_sequence)
    flag = True
    # if (dist_in.power_law.alpha < in_power * 0.75) or (dist_in.power_law.alpha > in_power * 0.75):
    #     flag = False
    if (dist_out.power_law.alpha < out_power * 0.75) or (dist_out.power_law.alpha > out_power * 1.25):
        flag = False
    return flag, dist_in.power_law.alpha, dist_out.power_law.alpha


def generate_vascular_networks(
        json_rec, name_networks, gamma_space, repetitions, in_power, out_power
):
    for num_nodes in json_rec['num_nodes']:
        rep = 1
        for gamma_ in gamma_space:
            for _ in range(repetitions):
                if rep >= 15:
                    break
                try:
                    g = nx.random_powerlaw_tree(
                        n=num_nodes,
                        gamma=gamma_,
                        tries=1000
                    )
                except NetworkXError:
                    # print("This run failed. Starting another run! ")
                    continue
                else:
                    print("Succeed!")
                g = nx.DiGraph([(u, v) for (u, v) in g.edges() if u < v])
                flag, in_power_v, out_power_v = check_power_directed_graph(g, in_power, out_power)
                if flag:
                    print("Found rep ", rep)
                    print(
                        "With gamma: ", gamma
                    )
                    file_name = './' + name_networks + '/n' + str(num_nodes) + 'r' + str(rep)
                    save_graph(g, file_name, in_power_v, out_power_v)
                    rep += 1


if __name__ == "__main__":
    total_reps = 1500

    search_intervals = 90
    gamma_min = 1.5
    gamma_max = 4.9
    gamma = np.linspace(gamma_min, gamma_max, search_intervals)

    name_space = 'vascular_networks'
    with open('networks_json_data.json') as json_file:
        json_record = json.load(json_file)
    print(json_record)
    print('-' * 15)
    spec_json_record = json_record[name_space]
    print(spec_json_record)

    generate_vascular_networks(
        json_rec=spec_json_record,
        name_networks=name_space,
        gamma_space=gamma,
        repetitions=total_reps,
        in_power=3.7,
        out_power=3.8
    )

# g = nx.binomial_tree(n=4, create_using=nx.DiGraph)

# g = nx.binomial_graph(
#     n=10,
#     p=0.3,
#     directed=True
# )

#
# print("Number of nodes: ", len(g.nodes))
# print("In-degree: ", list(d for n, d in g.in_degree()))
# print("out-degree: ", list(d for n, d in g.out_degree()))
# print("In-degree power: ", in_power)
# print("Out-degree power: ", out_power)
# print("Density: ", nx.density(g))
# print("Assortativity: ", nx.degree_assortativity_coefficient(g))
# print("Clustering Coefficient: ", nx.average_clustering(g))
# print("Average Shortest Path Length: ", nx.average_shortest_path_length(g))
# plot_graph(g)

