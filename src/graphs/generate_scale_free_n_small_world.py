import networkx as nx
import numpy as np
import json
import math
import matplotlib.pyplot as plt


def property_check(nx_graph, json_rec):
    degree_sequence = sorted((d for n, d in nx_graph.degree()), reverse=True)
    nx_graph = nx.DiGraph(nx_graph)
    und_graph = nx_graph.to_undirected()
    if not nx.is_connected(und_graph):
        return False
    if json_rec['clustering_c']:
        nx_c = nx.average_clustering(nx_graph)
        # print("clustering_c: ", nx_c)
        if nx_c > json_rec['clustering_c_max']:
            return False
        if nx_c < json_rec['clustering_c_min']:
            return False
    if json_rec['avg_shortest_path']:
        if not nx.is_strongly_connected(nx_graph):
            return False
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
            return False
        if nx_c < json_rec['density_min']:
            return False
    # if json_rec['directed']:
    #     nx_c = nx.density(nx_graph)
    #     if nx_c > json_rec['density_max']:
    #         return False
    #     if nx_c < json_rec['density_min']:
    #         return False
    return True


def get_adjacency(nx_graph):
    return np.array(nx.adjacency_matrix(nx_graph).todense())


def check_attributes(nx_graph):
    """
    Check if the graph contains node attributes
    :param nx_graph:
    :return:
    """
    for node in nx_graph.nodes:
        node_dict = nx_graph.nodes[node]
        if len(node_dict) != 0:
            return True
    return False


def remove_node_attributes(nx_graph):
    """
    Check if the graph contains node attributes
    :param nx_graph:
    :return:
    """
    g_c = nx_graph.copy()
    for node in nx_graph.nodes:
        node_dict = nx_graph.nodes[node]
        if len(node_dict) != 0:
            for key in node_dict:
                del g_c.nodes[node][key]
    return g_c


def save_graph(nx_graph, name_string,):
    adj = get_adjacency(nx_graph)
    name_adj = name_string + '_adj.npy'
    name_graphml = name_string + '.graphml'
    print("name string: ", name_string)
    name_plot = name_string + '.png'

    print(type(nx_graph))

    degree_sequence = sorted((d for n, d in nx_graph.degree()), reverse=True)
    dmax = max(degree_sequence)

    # save adjacency matrix as .npy
    np.save(name_adj, adj)

    if not check_attributes(nx_graph):
        # save the nx graph as .graphml:
        nx.write_graphml(nx_graph, name_graphml)
    else:
        nx_graph = remove_node_attributes(nx_graph)
        nx.write_graphml(nx_graph, name_graphml)

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
    plt.savefig(name_plot)
    plt.clf()


def watts_strogatz_n_b_a_graph(n_nodes, k, p, m):
    G1 = nx.newman_watts_strogatz_graph(n=int(n_nodes / 2), k=int(k), p=p)
    # print("m: ", m)
    G = nx.barabasi_albert_graph(n=n_nodes, m=int(m), initial_graph=G1)
    return G


def deterministic_insertion_growth_model(n_nodes):
    edge_list = []
    for i in range(n_nodes):
        edge_list.append([])
    initial_id = 0
    num_nodes_appended = 2

    return


def generate_scale_free_n_small_world_networks(
        json_rec, name_networks, p_prob_space, k_neighbors_space, m_edges_space, repetitions
):
    for num_nodes in json_rec['num_nodes']:
        print("Working on graphs with ", num_nodes, " nodes.")
        rep = 1
        for p_prob_i in p_prob_space:
            for k_neighbors_i in k_neighbors_space:
                if k_neighbors_i >= int(num_nodes/2):
                    continue
                for m_edges_i in m_edges_space:
                    for _ in range(repetitions):
                        if rep >= 15:
                            break
                        G = watts_strogatz_n_b_a_graph(num_nodes, k_neighbors_i, p_prob_i, m_edges_i)
                        if property_check(G, json_rec):
                            print("Found rep ", rep)
                            print("With prob: ", p_prob_i, ", m_edges: ", m_edges_i, ", k_neighbors: ", k_neighbors_i)
                            file_name = './' + name_networks + '/n' + str(num_nodes) + 'r' + str(rep)
                            print("Save at: " + file_name)
                            save_graph(G, file_name)
                            rep += 1


def generate_scale_free_n_small_world_networks_d(
        json_rec, name_networks, p_prob_space, k_neighbors_space, m_edges_space
):
    for num_nodes in json_rec['num_nodes']:
        rep = 1
        for p_prob_i in p_prob_space:
            for k_neighbors_i in k_neighbors_space:
                for m_edges_i in m_edges_space:
                    G = watts_strogatz_n_b_a_graph(num_nodes, k_neighbors_i, p_prob_i, m_edges_i)
                    if property_check(G, json_rec):
                        print("Found rep ", rep)
                        print("With prob: ", p_prob_i, ", m_edges: ", m_edges_i, " k_neighbors: ", k_neighbors_i)
                        file_name = './' + name_networks + 'n' + str(num_nodes) + 'r' + str(rep)
                        print("Save at: " + file_name)
                        save_graph(G, file_name)
                        rep += 1


if __name__ == '__main__':
    total_rep = 1500
    # search space:
    search_intervals = 45
    k_neighbors_min = 2
    k_neighbors_max = 8
    p_prob_min = 0.1
    p_prob_max = 0.7
    m_edges_min = 1
    m_edges_max = 5
    k_neighbors = np.linspace(k_neighbors_min, k_neighbors_max, search_intervals)
    p_prob = np.linspace(p_prob_min, p_prob_max, search_intervals)
    m_edges = np.linspace(m_edges_min, m_edges_max, search_intervals)

    name_space = 'gene_coexpression_networks'
    with open('networks_json_data.json') as json_file:
        json_record = json.load(json_file)

    print(json_record)
    print('-' * 15)
    spec_json_record = json_record[name_space]
    print(spec_json_record)
    generate_scale_free_n_small_world_networks(
        json_rec=spec_json_record,
        name_networks=name_space,
        p_prob_space=p_prob,
        k_neighbors_space=k_neighbors,
        m_edges_space=m_edges,
        repetitions=total_rep
    )
