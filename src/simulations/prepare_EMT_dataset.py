# import ot
import numpy as np
import anndata
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib as mpl
import wot
import scipy
import requests
import zipfile
import os
import networkx as nx
from io import StringIO

# download source data from https://github.com/yutongo/TIGON/
with open("./emt36_grn_network/AE_EMT_normalized.csv.zip", "wb") as f:
    f.write(requests.get("https://github.com/yutongo/TIGON/raw/main/EMT_data/AE_EMT_normalized.csv.zip").content)
with open("./emt36_grn_network/AE_EMT_time.csv", "wb") as f:
    f.write(requests.get("https://github.com/yutongo/TIGON/raw/main/EMT_data/AE_EMT_time.csv").content)

# unzip the data and remove redundant files
with zipfile.ZipFile("./emt36_grn_network/AE_EMT_normalized.csv.zip", 'r') as zip_ref:
    zip_ref.extract("./emt36_grn_network/AE_EMT_normalized.csv")
os.remove("./emt36_grn_network/AE_EMT_normalized.csv.zip")

# read the files
adata = anndata.AnnData(X=pd.read_csv("./emt36_grn_network/AE_EMT_normalized.csv", index_col=0).T, obs=pd.read_csv("./emt36_grn_network/AE_EMT_time.csv", index_col=0))
adata.obs = adata.obs["x"].map({
    "8h": 1/3,
    "1d": 1,
    "3d": 3,
    "0d": 0,
    "7d": 7,
}).to_frame(name="time")
adata_hvg = adata[:,np.argsort(adata.X.std(axis=0))[-100:]]

def get_ref_net_real_data(data):
    gene_list = data.var.index.values.tolist()

    string_db_query = requests.post("https://string-db.org/api/tsv/network", data = {
                        "identifiers": "%0d".join(gene_list),
                        "species": 9606,
                        "show_query_node_labels": 1
                    })
    string_result_df = pd.read_csv(StringIO(string_db_query.text), sep="\t").drop_duplicates()
    # string_result_df["ed_score"] = 1-(1-string_result_df[["escore", "dscore"]]).prod(axis = 1)
    # string_result_df = string_result_df[string_result_df["ed_score"] >= 0.4]
    adj = np.zeros((len(gene_list),len(gene_list)))
    for i in range(string_result_df.shape[0]):
        # adj[gene_list.index(string_result_df.iloc[i,2]), gene_list.index(string_result_df.iloc[i,3])] = 1
        adj[gene_list.index(string_result_df.iloc[i,2]), gene_list.index(string_result_df.iloc[i,3])] = string_result_df.iloc[i,5]
    adj = adj + adj.T
    return(pd.DataFrame(adj, index=gene_list, columns=gene_list))

ref_net = get_ref_net_real_data(adata_hvg)
idx_50 = list(next(nx.connected_components(nx.from_numpy_array(ref_net.values[:50,:50]))))
adata = adata_hvg[:,idx_50]
# adata_hvg[:,idx_50].write_h5ad("EMT_hvg50cc36.h5ad")
# adata_hvg[:,idx_50].to_df().to_csv("EMT50cc36.csv")
# idx_100 = list(next(nx.connected_components(nx.from_numpy_array(ref_net.values))))
# adata_hvg[:,idx_100].write_h5ad("EMT_hvg100cc83.h5ad")
# adata_hvg[:,idx_100].to_df().to_csv("EMT100cc83.csv")

adata = adata[np.argsort(adata.obs['time']),:]

timepoints = adata.obs['time'].value_counts().sort_index()

def sampling_path(transition_matrixes, num_trajectory=None):
    assert len(transition_matrixes) == len(timepoints)-1
    t0_cell_num = timepoints.values[0]
    if num_trajectory is None:
        num_trajectory = t0_cell_num
    out = np.zeros((num_trajectory, len(timepoints)), dtype=int)
    out[:num_trajectory//t0_cell_num * t0_cell_num, 0] = np.repeat(np.arange(t0_cell_num), num_trajectory//t0_cell_num)
    out[num_trajectory//t0_cell_num * t0_cell_num:, 0] = np.arange(num_trajectory%t0_cell_num)
    index_offset = 0
    for i in range(len(timepoints)-1):
        start_idx = out[:,i] - index_offset
        index_offset += timepoints.values[i]
        # end_idx = (transition_matrixes[i][start_idx,:] * np.random.random((num_trajectory, transition_matrixes[i].shape[1]))).argmax(axis=1)
        end_idx = transition_matrixes[i][start_idx,:].argmax(axis=1)
        out[:,i+1] = end_idx + index_offset
    return(out)

# transition_matrixes = [
#     ot.unbalanced.sinkhorn_unbalanced(
#                         a = np.ones(timepoints.values[i]),
#                         b = np.ones(timepoints.values[i+1]),
#                         M = ot.dist(adata[adata.obs['time'] == timepoints.index[i],:].X, adata[adata.obs['time'] == timepoints.index[i+1],:].X),
#                         reg = 1e0,
#                         # reg_m = np.inf,
#                         reg_m = 1e0,
#                         reg_type="entropy"
#                     ) for i in range(len(timepoints)-1)
# ]
ot_model = wot.ot.OTModel(adata, epsilon = 0.05, lambda1 = 1, lambda2 = 50, day_field="time")
transition_matrixes = [
    ot_model.compute_transport_map(timepoints.index[i], timepoints.index[i+1]).X
    for i in range(len(timepoints)-1)
]

sample_paths = sampling_path(transition_matrixes)

sim_timepoints = np.linspace(timepoints.index[0], timepoints.index[-1], 22).tolist()
traj = np.zeros((len(sample_paths), len(sim_timepoints), adata.shape[1]))
traj[:,0,:] = adata.X[sample_paths[:,0],:]
for i in range(len(timepoints)-1):
    start_cell = adata.X[sample_paths[:,i],:]
    end_cell = adata.X[sample_paths[:,i+1],:]
    traj[:,sim_timepoints.index(timepoints.index[i+1]),:] = end_cell
    diff = sim_timepoints.index(timepoints.index[i+1]) - sim_timepoints.index(timepoints.index[i])
    if diff > 1:
        # traj[:,sim_timepoints.index(timepoints.index[i])+1:sim_timepoints.index(timepoints.index[i+1]),:] = (start_cell + np.multiply.outer(np.linspace(0, 1, diff+1)[1:-1], end_cell - start_cell)).transpose(1,0,2)
        for j in range(1, diff):
            np.random.seed(0)
            traj[:,sim_timepoints.index(timepoints.index[i])+j,:] = wot.ot.interpolate_with_ot(start_cell,
                                                                                               end_cell,
                                                                                               transition_matrixes[i][sample_paths[:,i]-timepoints.values[:i].sum(),:][:,sample_paths[:,i+1]-timepoints.values[:i+1].sum()],
                                                                                               np.linspace(0, 1, diff+1)[j],
                                                                                               len(sample_paths))

sim_timepoints = np.linspace(timepoints.index[0], timepoints.index[-1], 22).tolist()
traj = np.zeros((len(sample_paths), len(sim_timepoints), adata.shape[1]))
for gene_id in range(adata.shape[1]):
    interpolator = scipy.interpolate.PchipInterpolator(timepoints.index, adata.X[sample_paths.reshape(-1),gene_id].reshape(sample_paths.shape), axis=1)
    traj[:,:,gene_id] = interpolator(sim_timepoints)

for t_idx, t in enumerate(timepoints.index):
    assert np.allclose(traj[:,sim_timepoints.index(t),:], adata.X[sample_paths[:,t_idx],:])

np.save("./emt36_grn_network/traj_EMT50cc36_wot_interpolated_by_pchip_22t.npy", traj)
np.save("./emt36_grn_network/edges_EMT36.npy", ref_net.values[:,idx_50][idx_50,:])