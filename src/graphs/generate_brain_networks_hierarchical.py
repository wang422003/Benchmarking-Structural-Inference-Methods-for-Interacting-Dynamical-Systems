import numpy as np
import networkx as nx
import random
import json
from generate_scale_free_n_small_world import property_check
from generate_intercellular_networks import save_graph
from collections import Counter


def duplicate_subgraph(num_repetitions, edge_list, max_id_so_far):
    edge_list = np.array(edge_list)
    max_id = max_id_so_far
    sub_g = nx.DiGraph()
    for _ in range(num_repetitions):
        edge_list_this_round = edge_list + max_id
        max_id = edge_list_this_round.max()
        sub_g.add_edges_from(edge_list_this_round)
    return sub_g, max_id


def duplicate_subgraph_fast(num_repetitions, edge_list, max_id_so_far=0):
    edge_list = np.array(edge_list)
    # print("edge_list:")
    # print(edge_list)
    sub_g = nx.DiGraph()
    for _ in range(num_repetitions):
        cache_g = nx.DiGraph((x[0], x[1]) for x in edge_list)
        # cache_g = nx.DiGraph(edge_list)
        sub_g = nx.disjoint_union(cache_g, sub_g)
    return sub_g, max_id_so_far


def total_nodes_combination(node_counts, num_nodes, freq_list):
    max_nodes = np.array(node_counts).max()
    all_combi = list()
    for i in range(num_nodes // max_nodes):
        if i == 0:
            continue
        all_combi.append(np.array(random.choices(list(enumerate(node_counts)), k=i, weights=freq_list))[:, 0])
    return np.array(all_combi)


def combine_components(nx_digraph):
    """
    Combine the disconnected components in the directed graph with k-edge-augmentation
    the inserted connections are bi-directed
    :param nx_digraph: a nx directed graph
    :return: a nx directed graph
    """
    nx_graph = nx_digraph.to_undirected()
    complement = list(nx.k_edge_augmentation(nx_graph, k=1))
    complement_rev = [i[::-1] for i in complement]
    nx_digraph.add_edges_from(complement)
    nx_digraph.add_edges_from(complement_rev)
    return nx_digraph


def sample_subgraphs(a_combination, edge_list_all):
    combi_dict = {i: np.count_nonzero(a_combination == i) for i in a_combination}
    max_id = 0
    sum_g = nx.DiGraph()
    for key in combi_dict:  # following fast subgraph call
        sub_g, max_id = duplicate_subgraph_fast(
            num_repetitions=combi_dict[key],
            edge_list=edge_list_all[key],
            max_id_so_far=max_id
        )
        sum_g = nx.disjoint_union(sum_g, sub_g)
    # connect components with bi-directed connections
    sum_g = combine_components(sum_g)
    return sum_g


def expand_graph(nx_graph, num_nodes, alpha, beta, gamma, delta_in, delta_out):
    """
    Expand the graph to the desired number of nodes with scale-free-graph
    :param nx_graph: a nx DiGraph
    :param num_nodes: number of nodes desired
    :return: a nx DiGraph
    """
    num_nodes_ingraph = len(nx_graph.nodes)
    if num_nodes_ingraph == num_nodes:
        return nx_graph
    nx_graph = nx.scale_free_graph(
        n=num_nodes,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta_in=delta_in,
        delta_out=delta_out,
        initial_graph=nx.MultiDiGraph(nx_graph)
    )
    return nx_graph


def generate_graph_with_hierarchical(json_rec, name_network, repetitions):

    for num_nodes in json_rec['num_nodes']:
        rep = 1
        small_counts = int(num_nodes / 5)
        for _ in range(repetitions):
            if rep >= 15:
                break
            new_edge_list = []
            for i in range(small_counts):
                g_small = nx.gn_graph(n=5, create_using=nx.DiGraph())
                # print("edge_list: ")
                for line in nx.generate_edgelist(g_small, data=False):
                    a = int(line[0]) + i * 5
                    b = int(line[-1]) + i * 5

                    new_edge_list.append((a, b))
            g = nx.DiGraph()
            g.add_edges_from(new_edge_list)

            und_g = g.to_undirected()
            complement = nx.k_edge_augmentation(und_g, k=1)
            # print(complement)

            g.add_edges_from(complement)
            print("Found rep ", rep)
            file_name = './' + name_network + '/n' + str(num_nodes) + 'r' + str(rep)
            print("Save at: " + file_name)
            save_graph(g, file_name)
            rep += 1



if __name__ == "__main__":
    total_rep = 150

    name_space = 'brain_networks'
    with open('networks_json_data.json') as json_file:
        json_record = json.load(json_file)
    print(json_record)
    print('-' * 15)
    spec_json_record = json_record[name_space]
    print(spec_json_record)

    generate_graph_with_hierarchical(
        json_rec=spec_json_record,
        name_network=name_space,
        repetitions=total_rep
    )
