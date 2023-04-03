import numpy as np
import torch
from tqdm import tqdm
from sklearn import feature_selection
import glob
import re


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def mi_terms(edges, latent, num_nodes):
    mi_sum = 0.0
    for i in tqdm(range(edges.shape[1])):
        mi = feature_selection.mutual_info_regression(X=edges[:, i, :], y=latent[:, i], copy=True)
        # print(i)
        # print(np.sum(mi))
        mi_sum += mi
    return np.sum(mi_sum) / (num_nodes ** 2)


def n2e_information(node_features, num_nodes):
    structure = np.ones((num_nodes, num_nodes))
    rel_rec = np.array(encode_onehot(np.where(structure)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(structure)[1]), dtype=np.float32)
    receivers = np.matmul(rel_rec, node_features)
    senders = np.matmul(rel_send, node_features)
    d0, d1, d2 = receivers.shape
    edges = np.concatenate([senders[:, :, :, np.newaxis], receivers[:, :, :, np.newaxis]], axis=-1)
    edges = np.reshape(edges, (d0, d1, 2 * d2))
    return edges


def mi_calculation(probs_folder_path, num_nodes, suffix):
    node_data = np.load('./Synthetic-H/sampled_data/' + suffix + '/train.npy')
    n_batch = node_data.shape[0]
    edges_in = n2e_information(node_data, num_nodes)
    # edges_out = np.concatenate([edges_in[:, 1:, :], edges_in[:, -1, :][:, np.newaxis, :]], axis=1)
    edges_out = edges_in[:, 1:, :]
    edges_in = edges_in[:, :-1, :]
    reps_list = []
    mi_xz_list = []
    mi_zy_list = []
    for file in glob.glob(probs_folder_path + '*.npy'):
        probs = np.load(file)
        reps = probs.shape[0] / n_batch

        for i in range(int(reps)):
            mi_xz = mi_terms(
                edges=edges_in,
                latent=probs[(i * n_batch): ((i + 1) * n_batch), :],
                num_nodes=num_nodes
            )
            mi_zy = mi_terms(
                edges=edges_out,
                latent=probs[(i * n_batch): ((i + 1) * n_batch), :],
                num_nodes=num_nodes
            )
            mi_xz_list.append(mi_xz)
            mi_zy_list.append(mi_zy)
        reps_list.append(reps)
    mi_xz_list = np.array(mi_xz_list)
    mi_zy_list = np.array(mi_zy_list)
    reps_list = np.array(reps_list)
    save_mi_xz = probs_folder_path + 'mi_xz.npy'
    save_mi_zy = probs_folder_path + 'mi_zy.npy'
    save_reps = probs_folder_path + 'reps_list.npy'
    np.save(save_mi_xz, mi_xz_list)
    np.save(save_mi_zy, mi_zy_list)
    np.save(save_reps, reps_list)
    print('Mutual information calculated and stored at ' + probs_folder_path)


def mi_calculation_name_order(probs_folder_path, num_nodes, suffix, off_d=False, sample_epochs=True):
    """
    Order the file names from glob.glob and then calculate mutual information.
    The mutual information is calculated based on sampling with fixed distance
    Sample 100 batches from the epoch and sample 100 epochs from the data
    :param sample_epochs:
    :param off_d:
    :param probs_folder_path: the path of the saved weight (not the "probs" folder)
    :param num_nodes:
    :param suffix:
    :return:
    """

    reps_list = []
    mi_xz_list = []
    mi_zy_list = []
    li_new = []

    node_data = np.load('./Synthetic-H/sampled_data/' + suffix + '/train.npy')
    n_batch = node_data.shape[0]
    edges_in = n2e_information(node_data, num_nodes)
    # edges_out = np.concatenate([edges_in[:, :, 2:], edges_in[:, :, -2:][:, np.newaxis, :]], axis=1)
    edges_out = np.concatenate([edges_in[:, :, 2:], edges_in[:, :, -2:]], axis=-1)

    if off_d:
        select = np.linspace(0, num_nodes ** 2 - 1, num_nodes ** 2)
        off_diag = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)
        off_diag = off_diag.flatten()
        select = select[np.where(off_diag)]
        select = select.astype(int)
        print(select)
        edges_in = edges_in[:, select, :]
        edges_out = edges_out[:, select, :]

    if n_batch > 100:
        batch_ind = np.linspace(0, n_batch - 1, 100, dtype=int)
        edges_out = edges_out[batch_ind, :, :]
        edges_in = edges_in[batch_ind, :, :]
        n_batch = len(batch_ind)
    else:
        batch_ind = np.linspace(0, n_batch - 1)

    li = glob.glob(probs_folder_path + '*.npy')
    li.sort(key=natural_keys)

    n_epochs = len(li)
    if n_epochs > 100 and sample_epochs:
        epoch_ind = np.linspace(0, n_epochs - 1, 100, dtype=int)
        for idx in epoch_ind:
            li_new.append(li[idx])
    else:
        li_new = li

    for file in li_new:
        probs = np.load(file)
        probs = probs[batch_ind, :]
        reps = probs.shape[0] / 100

        for i in range(int(reps)):
            mi_xz = mi_terms(
                edges=edges_in,
                latent=probs[(i * n_batch): ((i + 1) * n_batch), :],
                num_nodes=num_nodes
            )
            mi_zy = mi_terms(
                edges=edges_out,
                latent=probs[(i * n_batch): ((i + 1) * n_batch), :],
                num_nodes=num_nodes
            )
            mi_xz_list.append(mi_xz)
            mi_zy_list.append(mi_zy)
        reps_list.append(reps)
    mi_xz_list = np.array(mi_xz_list)
    mi_zy_list = np.array(mi_zy_list)
    reps_list = np.array(reps_list)
    save_mi_xz = probs_folder_path + 'mi_xz.npy'
    save_mi_zy = probs_folder_path + 'mi_zy.npy'
    save_reps = probs_folder_path + 'reps_list.npy'
    np.save(save_mi_xz, mi_xz_list)
    np.save(save_mi_zy, mi_zy_list)
    np.save(save_reps, reps_list)
    print('Mutual information calculated and stored at ' + probs_folder_path)


if __name__ == '__main__':
    print('THIS IS MUTUAL INFORMATION CALCULATION!')
