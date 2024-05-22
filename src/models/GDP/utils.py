import pickle
import torch
import os.path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import matplotlib.pyplot as plt
from typing import Optional

def portion_data(raw_data, data_portion, time_steps, shuffle):
    if data_portion == 1.0 and time_steps == 49:
        return raw_data
    if shuffle:
        np.random.shuffle(raw_data)
    num_trajs = raw_data.shape[0]
    num_times = raw_data.shape[0]
    return raw_data[:int(num_trajs * data_portion), :int(time_steps), :, :]

def load_customized_springs_data(args):

    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'

    loc_train = np.load(root_str + 'loc_train_' + keep_str)
    vel_train = np.load(root_str + 'vel_train_' + keep_str)
    edges_train = np.load(root_str + 'edges_train_' + keep_str)
    edges_train[edges_train > 0] = 1

    loc_valid = np.load(root_str + 'loc_valid_' + keep_str)
    vel_valid = np.load(root_str + 'vel_valid_' + keep_str)
    edges_valid = np.load(root_str + 'edges_valid_' + keep_str)
    edges_valid[edges_valid > 0] = 1

    loc_test = np.load(root_str + 'loc_test_' + keep_str)
    vel_test = np.load(root_str + 'vel_test_' + keep_str)
    edges_test = np.load(root_str + 'edges_test_' + keep_str)
    edges_test[edges_test > 0] = 1

    assert np.allclose(edges_train, edges_test) and np.allclose(edges_train, edges_valid), 'Edges are not consistent'
    edges = edges_train

    loc_train = portion_data(loc_train, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_train = portion_data(vel_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    loc_valid = portion_data(loc_valid, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_valid = portion_data(vel_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    num_nodes = loc_train.shape[3]

    n_train = loc_train.shape[0]
    n_test = loc_test.shape[0]
    n_valid = loc_valid.shape[0]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    feat_train = np.concatenate([loc_train, vel_train], axis=2)
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=2)
    feat_test = np.concatenate([loc_test, vel_test], axis=2)

    # Reshape to: [num_sims, num_timesteps, num_nodes, num_dims]
    # loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    # vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    # feat_train = np.concatenate([loc_train, vel_train], axis=3)
    # edges_train = np.tile(edges_train, (n_train, 1, 1))
    # edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])

    # loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    # vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    # feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    # edges_valid = np.tile(edges_valid, (n_valid, 1, 1))
    # edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])

    # loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    # vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    # feat_test = np.concatenate([loc_test, vel_test], axis=3)
    # edges_test = np.tile(edges_test, (n_test, 1, 1))
    # edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    # feat_train = torch.FloatTensor(feat_train)
    # edges_train = torch.LongTensor(edges_train)
    # feat_valid = torch.FloatTensor(feat_valid)
    # edges_valid = torch.LongTensor(edges_valid)
    # feat_test = torch.FloatTensor(feat_test)
    # edges_test = torch.LongTensor(edges_test)

    # Exclude self edges: discarded
    # off_diag_idx = np.ravel_multi_index(
    #     np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
    #     [num_atoms, num_atoms])
    # edges_train = edges_train[:, off_diag_idx]
    # edges_valid = edges_valid[:, off_diag_idx]
    # edges_test = edges_test[:, off_diag_idx]

    # train_data = TensorDataset(feat_train, edges_train)
    # valid_data = TensorDataset(feat_valid, edges_valid)
    # test_data = TensorDataset(feat_test, edges_test)

    # train_data_loader = DataLoader(train_data, batch_size=args.batch_size)
    # valid_data_loader = DataLoader(valid_data, batch_size=args.batch_size)
    # test_data_loader = DataLoader(test_data, batch_size=args.batch_size)

    return feat_train, feat_valid, feat_test, edges


def load_customized_netsims_data(args):

    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'

    bold_train = np.load(root_str + 'bold_train_' + keep_str)
    edges_train = np.load(root_str + 'edges_train_' + keep_str)
    edges_train[edges_train > 0] = 1

    bold_valid = np.load(root_str + 'bold_valid_' + keep_str)
    edges_valid = np.load(root_str + 'edges_valid_' + keep_str)
    edges_valid[edges_valid > 0] = 1

    bold_test = np.load(root_str + 'bold_test_' + keep_str)
    edges_test = np.load(root_str + 'edges_test_' + keep_str)
    edges_test[edges_test > 0] = 1

    assert np.allclose(edges_train, edges_test) and np.allclose(edges_train, edges_valid), 'Edges are not consistent'
    edges = edges_train

    bold_train = portion_data(bold_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    bold_valid = portion_data(bold_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    # num_nodes = bold_train.shape[3]

    # n_train = bold_train.shape[0]
    # n_test = bold_test.shape[0]
    # n_valid = bold_valid.shape[0]

    bold_max = bold_train.max()
    bold_min = bold_train.min()

    # bold_train = np.log(np.abs(bold_train))*np.sign(bold_train)
    # bold_valid = np.log(np.abs(bold_valid))*np.sign(bold_valid)
    # bold_test = np.log(np.abs(bold_test))*np.sign(bold_test)
    
    bold_train = (bold_train - bold_min) * 2 / (bold_max - bold_min) - 1
    bold_valid = (bold_valid - bold_min) * 2 / (bold_max - bold_min) - 1
    bold_test = (bold_test - bold_min) * 2 / (bold_max - bold_min) - 1

    # bold_valid = (bold_valid - bold_valid.min()) * 2 / bold_valid.ptp() - 1
    # bold_test = (bold_test - bold_test.min()) * 2 / bold_test.ptp() - 1

    # Reshape to: [num_sims, num_timesteps, num_nodes, num_dims]
    # feat_train = np.transpose(bold_train, [0, 3, 1, 2])
    # edges_train = np.tile(edges_train, (n_train, 1, 1))
    # edges_train = np.reshape(edges_train, [-1, num_nodes ** 2])

    # feat_valid = np.transpose(bold_valid, [0, 3, 1, 2])
    # edges_valid = np.tile(edges_valid, (n_valid, 1, 1))
    # edges_valid = np.reshape(edges_valid, [-1, num_nodes ** 2])

    # feat_test = np.transpose(bold_test, [0, 3, 1, 2])
    # edges_test = np.tile(edges_test, (n_test, 1, 1))
    # edges_test = np.reshape(edges_test, [-1, num_nodes ** 2])

    # feat_train = torch.FloatTensor(feat_train)
    # edges_train = torch.LongTensor(edges_train)
    # feat_valid = torch.FloatTensor(feat_valid)
    # edges_valid = torch.LongTensor(edges_valid)
    # feat_test = torch.FloatTensor(feat_test)
    # edges_test = torch.LongTensor(edges_test)

    # Exclude self edges: discarded
    # off_diag_idx = np.ravel_multi_index(
    #     np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
    #     [num_atoms, num_atoms])
    # edges_train = edges_train[:, off_diag_idx]
    # edges_valid = edges_valid[:, off_diag_idx]
    # edges_test = edges_test[:, off_diag_idx]

    # train_data = TensorDataset(feat_train, edges_train)
    # valid_data = TensorDataset(feat_valid, edges_valid)
    # test_data = TensorDataset(feat_test, edges_test)

    # train_data_loader = DataLoader(train_data, batch_size=args.batch_size)
    # valid_data_loader = DataLoader(valid_data, batch_size=args.batch_size)
    # test_data_loader = DataLoader(test_data, batch_size=args.batch_size)

    return bold_train, bold_valid, bold_test, edges

# def load_data(args):
#     cur_dir = os.path.dirname(os.path.realpath(__file__))
#     data_path = './data/' + args.suffix +'.pickle'
#     data_path = os.path.join(cur_dir, data_path)

#     with open(data_path, 'rb') as f:
#         x_tr, x_va, x_te, A = pickle.load(f)
#     x_tr = torch.from_numpy(x_tr).to(torch.float32)
#     x_va = torch.from_numpy(x_va).to(torch.float32)
#     x_te = torch.from_numpy(x_te).to(torch.float32)
#     A = (np.rint(A)).astype(int)


#     # number of trajectories in train/validation/test set
#     if args.tr_num is not None:
#         assert x_tr.size(0) >= args.tr_num, 'No sufficent train data!'
#         x_tr = x_tr[:args.tr_num,...]
#     if args.va_num is not None:
#         assert x_va.size(0) >= args.va_num, 'No sufficent validation data!'
#         x_va = x_va[:args.va_num,...]
#     if args.te_num is not None:
#         assert x_te.size(0) >= args.te_num, 'No sufficent test data!'
#         x_te = x_te[:args.te_num,...]


#     # sample subsequence for larger obvervation interval
#     if args.sample_freq != 1:
#         x_tr = x_tr[:,0::args.sample_freq,:,:]
#         x_va = x_va[:,0::args.sample_freq,:,:]
#         x_te = x_te[:,0::args.sample_freq,:,:]



#     # trajectory length
#     if args.trajr_length is not None:
#         assert x_tr.size(1) >= args.trajr_length, 'Not enough trajectory length!' 
#         x_tr = x_tr[:,:args.trajr_length,:,:]
#         x_va = x_va[:,:args.trajr_length,:,:]
#         x_te = x_te[:,:args.trajr_length,:,:]

#     #Normalize each system state dimension to [-1, 1]
#     for i in range(x_tr.size(-1)):
#         xmax = x_tr[:,:,:,i].max()
#         xmin = x_tr[:,:,:,i].min()

#         x_tr[:,:,:,i] = (x_tr[:,:,:,i] - xmin) * 2 / (xmax - xmin) -1
#         x_va[:,:,:,i] = (x_va[:,:,:,i] - xmin) * 2 / (xmax - xmin) -1
#         x_te[:,:,:,i] = (x_te[:,:,:,i] - xmin) * 2 / (xmax - xmin) -1
    
#     # input data has shape [batch, time, nodes, variables]
#     x_tr = x_tr.permute(0,2,3,1)
#     x_va = x_va.permute(0,2,3,1)
#     x_te = x_te.permute(0,2,3,1)
#     # data has shape [batch, nodes, variables, time]

#     print('Training Trajectories: {:03d}'.format(x_tr.size(0)),
#         'Trajectory length: {:03d}'.format(x_tr.size(3)))

#     return x_tr, x_va, x_te, A



# def load_netsim_data(args,batch_size=1, datadir="data"):
#     print("Loading data from {}".format(datadir))

#     subject_id = [1, 2, 3, 4, 5]

#     print("Loading data for subjects ", subject_id)

#     loc_train = torch.zeros(len(subject_id), 15, 200)
#     edges_train = torch.zeros(len(subject_id), 15, 15)

#     for idx, elem in enumerate(subject_id):
#         fileName = "sim3_subject_%s.npz" % (elem)
#         ld = np.load(os.path.join(datadir, "netsim", fileName))
#         loc_train[idx] = torch.FloatTensor(ld["X_np"])
#         edges_train[idx] = torch.LongTensor(ld["Gref"])

#     # [num_sims, num_atoms, num_timesteps, num_dims]
#     loc_train = loc_train.unsqueeze(-1)

#     loc_max = loc_train.max()
#     loc_min = loc_train.min()
#     loc_train = normalize(loc_train, loc_min, loc_max)

#     if args.sample_freq != 1:
#         loc_train = loc_train[:,:,0::args.sample_freq]
    

#     loc_train = loc_train.permute(0,1,3,2)
    

#     x_tr = loc_train.clone()
#     x_va = loc_train.clone()
#     x_te = loc_train.clone()

#     # Exclude self edges
#     A = edges_train[0].int().numpy()
#     A = A - np.diag(np.diagonal(A))
#     print('Training Trajectories: {:03d}'.format(x_tr.size(0)),
#         'Trajectory length: {:03d}'.format(x_tr.size(3)))
#     return x_tr, x_va, x_te, A

def normalize(x, x_min, x_max):
    return (x - x_min) * 2 / (x_max - x_min) - 1


def cal_accuracy(A, A_soft, A_hard, num_edges, epoch=0):
    mask = ~np.eye(A.shape[0],dtype=bool)
    off_diag_idx = np.where(mask)

    scores = A_soft[off_diag_idx]
    labels = A[off_diag_idx]

    #auc
    auc = roc_auc_score(labels, scores)
    if auc < 0.5:
        auc = 1- auc 
        scores = - scores
        A_hard = 1-A_hard
    #acc
    acc = (labels == A_hard[off_diag_idx]).mean()

    ind = np.argsort(scores)
    pre = labels[ind[-num_edges:]].mean()

    return auc, acc, pre


class TrajrData(Dataset):
    def __init__(self, data, Tstep, interlacing=True):
        self.data = data.transpose(0,3,2,1)
        # data has shape [batch, nodes, variables, time]
        self.interlacing = interlacing
        if interlacing:
            self.Tout = self.data.shape[-1] - Tstep +1 #steps for reccurent output
        else:
            assert self.data.shape[-1]%Tstep == 0, 'Trajectory length must be integer multiple of Tstep'
            self.Tout = int(np.ceil(self.data.shape[-1]/Tstep))
        self.Tstep = Tstep
        self.batch = self.data.shape[0]
        self.datalen = self.batch*self.Tout
    def __len__(self):
        # return self.data.shape[0]
        return self.datalen
    def __getitem__(self, idx):
        i, j = idx//self.Tout, idx%self.Tout #i: batch, j: start time step
        if self.interlacing:
            sample = self.data[i,:,:,j:j+self.Tstep]
        else:
            start_ind = j*self.Tstep
            sample = self.data[i,:,:,start_ind:start_ind+self.Tstep]
        # sample = self.data[idx,:,:,0:self.Tstep]
        return sample

class TrajrData(Dataset):
    def __init__(self, data, Tstep, interlacing=True):
        self.data = data.transpose(0,3,2,1)
        # data has shape [batch, nodes, variables, time]
        self.interlacing = interlacing
        if interlacing:
            self.Tout = self.data.shape[-1] - Tstep +1 #steps for reccurent output
        else:
            assert self.data.shape[-1]%Tstep == 0, 'Trajectory length must be integer multiple of Tstep'
            self.Tout = int(np.ceil(self.data.shape[-1]/Tstep))
        self.Tstep = Tstep
        self.batch = self.data.shape[0]
        self.datalen = self.batch*self.Tout
    def __len__(self):
        return self.datalen
    def __getitem__(self, idx):
        i, j = idx//self.Tout, idx%self.Tout #i: batch, j: start time step
        if self.interlacing:
            sample = self.data[i,:,:,j:j+self.Tstep]
        else:
            start_ind = j*self.Tstep
            sample = self.data[i,:,:,start_ind:start_ind+self.Tstep]
        return sample

# def kl_categorical_uniform(
#     preds, num_atoms, add_const=False, eps=1e-16
# ):
#     """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
#     kl_div = preds * (torch.log(preds + eps))
#     if add_const:
#         const = np.log(num_edge_types)
#         kl_div += const
#     return kl_div.sum() / num_atoms

def nll_gaussian(preds, target, variance=5e-7, add_const=False):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    neg_log_p = (preds - target) ** 2 / (2 * variance)
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const

    return neg_log_p.sum() / (target.size(0) * target.size(1))


def generate_prediction(edge_prob, edge_index):
    A_hard = (edge_prob[1] > edge_prob[0]).long()
    A_soft = torch.softmax(0.5*edge_prob,dim=0)
    A_soft = to_dense_adj(edge_index, edge_attr = A_soft[1]).cpu().squeeze(0).numpy()
    A_hard = to_dense_adj(edge_index, edge_attr = A_hard).cpu().squeeze(0).numpy()
    return A_soft, A_hard

def generate_prediction_nri(edge_prob, edge_index):
    A_hard = (edge_prob[1] > edge_prob[0]).long()
    A_soft = torch.softmax(0.5*edge_prob,dim=0)
    A_soft = to_dense_adj(edge_index, edge_attr = A_soft[1]).cpu().squeeze(0).numpy()
    A_hard = to_dense_adj(edge_index, edge_attr = A_hard).cpu().squeeze(0).numpy()
    return A_soft, A_hard
   
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)