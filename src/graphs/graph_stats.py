import numpy as np
import networkx as nx
import os
import powerlaw
from generate_scale_free_n_small_world import check_attributes, remove_node_attributes, get_adjacency
import matplotlib.pyplot as plt

def save_graph(nx_graph, name_string,):
    name_ls = name_string.split('/')
    length = len(name_ls)
    new_name_st = ''
    for i in range(length):
        if i < length - 1:
            new_name_st += name_ls[i] + '/'
        else:
            new_name_st += 'cache/' + name_ls[i].split('.')[0]
    name_string = new_name_st
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

def check_power_directed_graph(nx_g):
    degree_sequence = sorted((d for n, d in nx_g.degree()), reverse=True)
    in_degree_sequence = sorted((d for n, d in nx_g.in_degree()), reverse=True)
    out_degree_sequence = sorted((d for n, d in nx_g.out_degree()), reverse=True)
    dist = powerlaw.Fit(degree_sequence)
    dist_in = powerlaw.Fit(in_degree_sequence)
    dist_out = powerlaw.Fit(out_degree_sequence)
    return dist.power_law.alpha, dist_in.power_law.alpha, dist_out.power_law.alpha

def check_power_undirected_graph(nx_g):
    degree_sequence = sorted((d for n, d in nx_g.degree()), reverse=True)
    dist = powerlaw.Fit(degree_sequence)
    return dist.power_law.alpha


def check_clustering_directed_one_by_one(nx_graph):
    nodes = list(nx_graph.nodes)
    cl_c = []
    for node in nodes:
        cl_c.append(nx.clustering(nx_graph, node))
    # print("clustering: ", cl_c)
    return np.array(cl_c).mean()
def property_check_directed(nx_graph):
    nx_n_nodes = 0
    nx_n_edges = 0
    nx_cl = 0
    nx_d = 0
    nx_k = 0
    nx_delta = 0
    nx_gamma = 0
    nx_gamma_in = 0
    nx_gamma_out = 0

    degree_sequence = sorted((d for n, d in nx_graph.degree()), reverse=True)
    nx_graph = nx.DiGraph(nx_graph)
    und_graph = nx_graph.to_undirected()

    nx_n_nodes = nx_graph.number_of_nodes()
    nx_n_edges = nx_graph.number_of_edges()
    if not nx.is_connected(und_graph):
        print("Not connected!")
        nx_cl = nx.average_clustering(und_graph)
    else:
        nx_cl = nx.average_clustering(nx_graph)
    if nx_cl == 0.0:
        print("Clustering is 0.0! Trying to calculate it one by one...")
        nx_cl = check_clustering_directed_one_by_one(nx_graph)
    # nx_cl = nx.average_clustering(und_graph)
    if not nx.is_strongly_connected(nx_graph):
        nx_d = nx.average_shortest_path_length(und_graph)
    else:
        nx_d = nx.average_shortest_path_length(nx_graph)
    nx_k = np.array(degree_sequence).mean()
    nx_delta = nx.density(nx_graph)
    nx_gamma, nx_gamma_in, nx_gamma_out = check_power_directed_graph(nx_graph)

    return nx_n_nodes, nx_n_edges, nx_cl, nx_d, nx_gamma, nx_k, nx_delta, nx_gamma_in, nx_gamma_out


def connect_undirected_graphs(nx_graph_, file_name):
    complement = list(nx.k_edge_augmentation(nx_graph_, k=1))
    complement_rev = [i[::-1] for i in complement]
    nx_graph_.add_edges_from(complement)
    save_graph(nx_graph_, file_name)
    return nx_graph_

def property_check_undirected(nx_graph, file_name):
    nx_n_nodes = 0
    nx_n_edges = 0
    nx_cl = 0
    nx_d = 0
    nx_k = 0
    nx_delta = 0
    nx_gamma = 0

    degree_sequence = sorted((d for n, d in nx_graph.degree()), reverse=True)
    und_graph = nx_graph.to_undirected()
    if not nx.is_connected(nx_graph):
        print("Not connected!")
        nx_graph = connect_undirected_graphs(nx_graph, file_name)
    nx_n_nodes = nx_graph.number_of_nodes()
    nx_n_edges = nx_graph.number_of_edges()
    nx_cl = nx.average_clustering(nx_graph)
    # if nx.is_connected(und_graph):
    nx_d = nx.average_shortest_path_length(nx_graph)
    # else:
    #     print("not connected!")
    nx_k = np.array(degree_sequence).mean()
    nx_delta = nx.density(nx_graph)
    nx_gamma = check_power_undirected_graph(nx_graph)

    return nx_n_nodes, nx_n_edges, nx_cl, nx_d, nx_gamma, nx_k, nx_delta


if __name__ == '__main__':
    directed = True # False

    folder_path = './brain_networks/'
    # folder_path = './chemical_reaction_networks_in_atmosphere/'
    # folder_path = './food_webs/'
    # folder_path = './gene_coexpression_networks/'
    # folder_path = './gene_regulatory_networks/'
    # folder_path = './intercellular_networks/'
    # folder_path = './landscape_networks/'
    # folder_path = './man-made_organic_reaction_networks/'
    # folder_path = './reaction_networks_inside_living_organism/'
    # folder_path = './social_networks/'
    # folder_path = './vascular_networks/'

    # be careful with the graphs that are directed or not
    if directed:
        actual_path = folder_path + 'ready/'
    else:
        actual_path = folder_path + 'un_ready/'

    file_list = []
    count = 0
    for file in os.listdir(actual_path):
        if file.endswith('.graphml'):
            count += 1
            # print(os.path.join(edges_dir_path, file))
            file_list.append(os.path.join(actual_path + file))

    file_list = sorted(file_list, key= lambda x: (int(x.split('/')[-1].split('n')[1].split('r')[0]),
                                                  int(x.split('/')[-1].split('n')[1].split('r')[1].split('.')[0])))
    for file in file_list:
        g = nx.read_graphml(file)
        if directed:
            nx_n_nodes_, nx_n_edges_, nx_cl_, nx_d_, nx_gamma_, nx_k_, nx_delta_, nx_gamma_in_, nx_gamma_out_ = property_check_directed(g)
            print("Graph: ", file)
            print("n_nodes: ", nx_n_nodes_, " n_edges: ", nx_n_edges_, "Cluster: ", nx_cl_, "Avg. path: ", nx_d_,
                  "gamma: ", nx_gamma_, "Avg. degree: ", nx_k_, "Density: ", nx_delta_,
                  "Gamma_in: ", nx_gamma_in_, "Gamma_out: ", nx_gamma_out_)
            print('-' * 25)
        else:
            nx_n_nodes_, nx_n_edges_, nx_cl_, nx_d_, nx_gamma_, nx_k_, nx_delta_ = property_check_undirected(g, file)
            print("Graph: ", file)
            print("n_nodes: ", nx_n_nodes_, " n_edges: ", nx_n_edges_, "Cluster: ", nx_cl_, "Avg. path: ", nx_d_,
                  "gamma: ", nx_gamma_, "Avg. degree: ", nx_k_, "Density: ", nx_delta_)
            print('-' * 25)
    print("Total number of graphs: ", count)
    print("Finished!")


