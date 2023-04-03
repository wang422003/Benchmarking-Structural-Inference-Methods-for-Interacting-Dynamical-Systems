"""
Undirected!
"""
import networkx as nx
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from generate_scale_free_n_small_world import property_check, save_graph
import powerlaw


def check_power(nx_graph, json_rec):
    if not nx.is_connected(nx_graph):
        return False
    degree_sequence = sorted((d for n, d in nx_graph.degree()), reverse=True)
    dist_all = powerlaw.Fit(degree_sequence)
    if dist_all.power_law.alpha < json_rec['gamma_min'] or dist_all.power_law.alpha > json_rec['gamma_max']:
        return False
    return True

def generate_gene_coexpression_networks(
        json_rec, name_networks, power_space, k_space, repetitions
):
    for num_nodes in json_rec['num_nodes']:
        print("Working on graphs with ", num_nodes, " nodes.")
        rep = 1
        for p_ in power_space:
            for _ in range(repetitions):
                # print("p: ", p_)
                if rep >= 15:
                    break
                s = []
                while True:
                    s = []
                    while len(s) < num_nodes:
                        nextval = int(nx.utils.powerlaw_sequence(1, p_)[0])  # 100 nodes, power-law exponent 2.5
                        if nextval != 0 and nextval < num_nodes:
                            s.append(nextval)
                    if sum(s) % 2 == 0:
                        break
                # print("s :", s)
                g = nx.configuration_model(s, create_using=nx.Graph())
                print(" graph generated, now check properties")
                if not nx.is_connected(g):
                    for k_ in k_space:
                        if rep >= 15:
                            break
                        complement = list(nx.k_edge_augmentation(g, k=int(k_), partial=True))
                        g.add_edges_from(complement)

                        # if property_check(g, json_rec) and check_power(g, json_rec):
                        if property_check_loose(g, json_rec):
                            print("Found rep ", rep)
                            print("With prob: ", p_, ", k_: ", int(k_))
                            file_name = './' + name_networks + '/n' + str(num_nodes) + 'r' + str(rep)
                            print("Save at: " + file_name)
                            save_graph(g, file_name)
                            rep += 1
                else:
                    # if property_check(g, json_rec) and check_power(g, json_rec):
                    if property_check_loose(g, json_rec):
                        print("Found rep ", rep)
                        print("With prob: ", p_)
                        file_name = './' + name_networks + '/n' + str(num_nodes) + 'r' + str(rep)
                        print("Save at: " + file_name)
                        save_graph(g, file_name)
                        rep += 1


def property_check_loose(nx_graph, json_rec):
    degree_sequence = sorted((d for n, d in nx_graph.degree()), reverse=True)
    # nx_graph = nx.DiGraph(nx_graph)
    und_graph = nx_graph.to_undirected()
    flag_c = 1
    flag_p = 1
    flag_d = 1
    if not nx.is_connected(und_graph):
        return False
    if json_rec['clustering_c']:
        nx_c = nx.average_clustering(nx_graph)
        # print("clustering_c: ", nx_c)
        if nx_c > json_rec['clustering_c_max']:
            flag_c = 0
        if nx_c < json_rec['clustering_c_min']:
            flag_c = 0
    if json_rec['avg_shortest_path']:
        # if not nx.is_strongly_connected(nx_graph):
        #     return False
        nx_c = nx.average_shortest_path_length(nx_graph)
        # print("avg_shortest_path: ", nx_c)
        if nx_c > json_rec['avg_shortest_path_max']:
            return False
        if nx_c < json_rec['avg_shortest_path_min']:
            return False
    if json_rec['avg_degree']:
        nx_c = np.array(degree_sequence).mean()
        # print("avg_degree: ", nx_c)
        if nx_c > json_rec['avg_degree_max']:
            return False
        if nx_c < json_rec['avg_degree_min']:
            return False
    if json_rec['density']:
        nx_c = nx.density(nx_graph)
        # print("density: ", nx_c)
        if nx_c > json_rec['density_max']:
            flag_d = 0
        if nx_c < json_rec['density_min']:
            flag_d = 0
    if json_rec['power_law']:
        p_check = check_power(nx_graph, json_rec)
        if not p_check:
            flag_p = 0
    # if json_rec['directed']:
    #     nx_c = nx.density(nx_graph)
    #     if nx_c > json_rec['density_max']:
    #         return False
    #     if nx_c < json_rec['density_min']:
    #         return False
    return (flag_c + flag_d) >= 1 and flag_p == 1


if __name__ == "__main__":
    total_rep = 150

    search_intervals = 30
    k_min = 1
    k_max = 10

    power_min = 1.4
    power_max = 2.8

    k = np.linspace(k_min, k_max, int(search_intervals / 2))
    power = np.linspace(power_min, power_max, search_intervals)

    name_space = 'gene_coexpression_networks'
    with open('networks_json_data.json') as json_file:
        json_record = json.load(json_file)

    print(json_record)
    print('-' * 15)
    spec_json_record = json_record[name_space]
    print(spec_json_record)

    generate_gene_coexpression_networks(
        json_rec=spec_json_record,
        name_networks=name_space,
        power_space=power,
        k_space=k,
        repetitions=total_rep
    )