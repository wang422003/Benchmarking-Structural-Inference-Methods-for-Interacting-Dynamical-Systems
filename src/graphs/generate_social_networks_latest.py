import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from generate_intercellular_networks import save_graph
from generate_scale_free_n_small_world import property_check


def property_check_loose(nx_graph, json_rec):
    degree_sequence = sorted((d for n, d in nx_graph.degree()), reverse=True)
    nx_graph = nx.DiGraph(nx_graph)
    und_graph = nx_graph.to_undirected()
    flag_c = 1
    flag_a = 1
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
        if not nx.is_strongly_connected(nx_graph):
            return False
        nx_c = nx.average_shortest_path_length(nx_graph)
        # print("avg_shortest_path: ", nx_c)
        if nx_c > json_rec['avg_shortest_path_max']:
            flag_a = 0
        if nx_c < json_rec['avg_shortest_path_min']:
            flag_a = 0
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
    # if json_rec['directed']:
    #     nx_c = nx.density(nx_graph)
    #     if nx_c > json_rec['density_max']:
    #         return False
    #     if nx_c < json_rec['density_min']:
    #         return False
    return (flag_c + flag_d + flag_a) >= 2

def generate_social_networks(
        json_rec, name_networks, p_space, repetitions, search_intervals_
):
    for num_nodes in json_rec['num_nodes']:
        if num_nodes > 15:
            continue
        rep = 1
        for p_v in p_space:
            for _ in range(repetitions):
                if rep >= 15:
                    break
                g = nx.gnp_random_graph(
                    n=num_nodes,
                    p=p_v,
                    directed=True
                )
                if property_check_loose(g, json_rec):
                    print("Found rep ", rep)
                    print(
                        "With p: ", p_v
                    )
                    file_name = './' + name_networks + '/n' + str(num_nodes) + 'r' + str(rep)
                    save_graph(g, file_name)
                    rep += 1


if __name__ == '__main__':
    total_rep = 600
    search_intervals = 990

    p_min = 0.01
    p_max = 0.99

    p = np.linspace(p_min, p_max, search_intervals)
    name_space = "social_networks"
    with open('networks_json_data.json') as json_file:
        json_record = json.load(json_file)
    print(json_record)
    print('-' * 15)
    spec_json_record = json_record[name_space]
    print(spec_json_record)
    generate_social_networks(
        json_rec=spec_json_record,
        name_networks=name_space,
        p_space=p,
        repetitions=total_rep,
        search_intervals_=search_intervals
    )
    print("FINISHED")

