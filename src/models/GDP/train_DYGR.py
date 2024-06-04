# import debugpy
# debugpy.listen(('0.0.0.0', 5679))
# debugpy.wait_for_client()

import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
import logging
import datetime
import time
import os

import torch
import torch.nn as nn
import torch.nn.utils as U
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from model_DYGR import *

t_begin = time.time()

parser = argparse.ArgumentParser()

# data
parser.add_argument('--suffix', type=str, default='netsims', help='suffix for data')
# parser.add_argument('--tr_num', type=int, default=None,
#     help='No. of training trajectories, using all trajectories when None')
# parser.add_argument('--va_num', type=int, default=None,
#     help='No. of validation trajectories, using all trajectories when None')
# parser.add_argument('--te_num', type=int, default=None,
#     help='No. of test trajectories, using all trajectories when None')
# parser.add_argument('--sample_freq', type=int, default=1,
#     help='Sampling frequency of the trajectory')
# parser.add_argument('--trajr_length', type=int, default=None,
#     help='No. of time stamps in each trajectory, using all data when None')
parser.add_argument('--interlacing', type=bool, default=True,
    help='If the trajectories are interlacing when preparing dataset')
parser.add_argument('--Tstep', type=int, default=2, help='No. of steps for batched trajectories')
# model
parser.add_argument('--skip_first_edge_type', type=bool, default=False,
    help='If skip non-edges')
parser.add_argument('--gumbel_noise', type=bool, default=False, help='If includes gumbel noise')
parser.add_argument('--beta', type=float, default=0.5, help='Inverse temperature in softmax function')
parser.add_argument('--init_logits', type=str, default='random',
    help='initialization of logtis, (uniform, random)')
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--heads', type=int, default=1, help='number of filters')
#parser.add_argument('--prior', type=float, default=0.01)
parser.add_argument('--prior', type=float, default=0.001)
# training
parser.add_argument('--lr', type=float, default=0.0005, 
    help="Initial learning rate.")
parser.add_argument('--lr_z', type=float, default=0.1, 
    help="Learning rate for distribution estimation.")
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--num_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--filter', type=str, default='cheby',
         help='polynomial filter type (cheby, power)')
parser.add_argument('--K', type=int, default=4,
    help='trucation in the order for polynomial filters') 
parser.add_argument('--num-layers', type=int, default=1, help='number of GCN layers')
parser.add_argument('--skip', type=bool, default=True,
                     help='wether to use the skip connection, if None then it will be infered from data')
parser.add_argument('--skip_poly', type=bool, default=False)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
# for benchmark:
# parser.add_argument('--save-probs', action='store_true', default=False,
#                     help='Save the probs during test.')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model.')
parser.add_argument('--checkpoint', type=str, default='',
                    help="Load the checkpoint file and take the file directory as save-folder.")
parser.add_argument('--b-portion', type=float, default=1.0,
                    help='Portion of data to be used in benchmarking.')
parser.add_argument('--b-time-steps', type=int, default=49,
                    help='Portion of time series in data to be used in benchmarking.')
parser.add_argument('--b-shuffle', action='store_true', default=False,
                    help='Shuffle the data for benchmarking?.')
# parser.add_argument('--b-manual-nodes', type=int, default=0,
#                     help='The number of nodes if changed from the original dataset.')
parser.add_argument('--data-path', type=str, default='',
                    help='Where to load the data. May input the paths to edges_train of the data.')
parser.add_argument('--b-network-type', type=str, default='',
                    help='What is the network type of the graph.')
parser.add_argument('--b-directed', action='store_true', default=False,
                    help='Default choose trajectories from undirected graphs.')
parser.add_argument('--b-simulation-type', type=str, default='',
                    help='Either springs or netsims.')
parser.add_argument('--b-suffix', type=str, default='',
    help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1.'
         ' Or "50r1" for 50 nodes, rep 1 and noise free.')
# remember to disable this for submission
parser.add_argument('--b-walltime', action='store_true', default=True,
                    help='Set wll time for benchmark training and testing. (Max time = 2 days)')
parser.add_argument('--cuda_device', type=int, default=None)

args = parser.parse_args()
# print(args)
# seed=args.seed
# if seed is None:
#     seed=random.randint(100,10000)
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

if args.suffix == "":
    args.suffix = args.b_simulation_type
    args.timesteps = args.b_time_steps

if args.b_simulation_type == 'springs':
    args.dims = 4
elif args.b_simulation_type == 'netsims':
    args.dims = 1

if args.data_path == "" and args.b_network_type != "":
    if args.b_directed:
        dir_str = 'directed'
    else:
        dir_str = 'undirected'
    args.data_path = '/project/scratch/p200352/bsimds/src/simulations/' + args.b_network_type + '/' + \
                     dir_str +\
                     '/' + args.b_simulation_type + '/edges_train_' + args.b_simulation_type + args.b_suffix + '.npy'
    # args.b_manual_nodes = int(args.b_suffix.split('r')[0])
# if args.data_path != '':
#     args.suffix = args.data_path.split('/')[-1].split('_', 2)[-1]
if args.checkpoint != '':
    checkpoint = args.checkpoint
    save_folder = os.path.dirname(args.checkpoint)
    args = torch.load(args.checkpoint)["args"]
    args.checkpoint = checkpoint
else:
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    save_folder = f"{args.save_folder}/exp{timestamp}_{random.randint(0, 1000)}/"
    os.mkdir(save_folder)

log_file = os.path.join(save_folder, 'log.txt')
logging.basicConfig(filename=log_file, filemode="a", level=logging.INFO, format='%(levelname)-8s :: %(asctime)s :: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if args.checkpoint == '':
    logger.info(args)
    
logger.info(f"suffix: {args.suffix}")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.cuda_device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else 'cpu')
if args.checkpoint == '':
    logger.info(f'Device: {device}')

if 'netsim' in args.suffix:
    x_tr, x_va, x_te, A = load_customized_netsims_data(args)
elif "springs" in args.suffix:
    x_tr, x_va, x_te, A = load_customized_springs_data(args) # loaded data has shape [batch, nodes, variables, time]

x_tr = x_tr.astype(np.float32)
x_va = x_va.astype(np.float32)
x_te = x_te.astype(np.float32)
A = A.astype(np.float32)

if args.K==0:
    args.K=1
    args.skip_poly=True


# args.Tstep = args.b_time_steps
Tstep = args.Tstep
batch_size = args.batch_size
epochs = args.num_epoch
lr = args.lr
num_nodes = A.shape[0]
num_variables = x_tr.shape[2]
num_edges = int(A.sum())
interlacing = args.interlacing
args.in_channels = num_variables
args.num_nodes = num_nodes
p=args.prior

train_loader = torch.utils.data.DataLoader(TrajrData(x_tr,Tstep,interlacing), batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(TrajrData(x_va,Tstep,interlacing), batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(TrajrData(x_te,Tstep,interlacing), batch_size=batch_size, shuffle=False)

if 'cheby' in args.filter:
    graph = ChebyGraphFilter(args).to(device)#.double()
elif 'power' in args.filter:
    graph = PowerGraphFilter(args).to(device)#.double()

dynamics = DynSurrogates(args).to(device)#.double()

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        nn.init.uniform_(m.bias.data, -1,1)

dynamics.apply(weights_init)
graph.apply(weights_init)

optimizer = torch.optim.Adam(
        [{"params": graph.parameters(), "lr": args.lr_z}] +
        [{"params": dynamics.parameters(), "lr": args.lr}]
    )

if args.checkpoint != '':
    ckpt = torch.load(args.checkpoint, map_location=device)
    graph.load_state_dict(ckpt['graph'])
    dynamics.load_state_dict(ckpt['dynamics'])
    optimizer.load_state_dict(ckpt['optimizer'])
    epochs = ckpt['epoch']+1
    
criterion = torch.nn.MSELoss(reduction='mean')

def train(data_loader):
    graph.train()
    dynamics.train()
    loss_batch = 0
    num_datum = 0

    for x in data_loader:
        x = x.to(device)
        edge_prob, filter_bank = graph(edge_index)
        xpreds = dynamics(x[...,:-1], edge_index, edge_prob, filter_bank)
        loss = 0
        for i in range(len(xpreds)):
            loss += criterion(xpreds[i],x[...,1:])
            num_datum += xpreds[i].numel()
        loss_batch += loss.item() * x[...,1:].numel()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_batch = loss_batch/num_datum
    return loss_batch

def test(data_loader):
    graph.eval()
    dynamics.eval()

    loss_batch = 0
    num_datum = 0
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            edge_prob, filter_bank = graph(edge_index)
            xpreds = dynamics(x[...,:-1], edge_index, edge_prob, filter_bank)
            loss = 0                                           
            for i in range(len(xpreds)):
                loss += criterion(xpreds[i],x[...,1:])
                num_datum += xpreds[i].numel()
            loss_batch += loss.item() * x[...,1:].numel()
    loss_batch = loss_batch/num_datum
    return loss_batch



if __name__ == '__main__':

    A_full = torch.ones(num_nodes,num_nodes)
    A_full = A_full - torch.diag(torch.diagonal(A_full))
    edge_index, _ = dense_to_sparse(A_full)
    edge_index = edge_index.to(device)

    best_val_loss = np.inf
    best_train_loss = np.inf
    best_mes_from_va = 0
    best_auc_from_va = 0
    best_acc_from_va = 0 
    loss_sum = 0 
    if args.checkpoint == '':
        epochs = 1
    for epoch in range(epochs, args.num_epoch+1):
        t_start = time.time()
        loss_tr = train(train_loader)
        if np.isfinite(loss_tr) == False:
            logger.error('Loss is not finite. Exiting training.')
            raise ValueError('Loss is not finite. Exiting training.')

        if(epoch%1 == 0):
            loss_va = test(valid_loader)
            loss_te = test(test_loader)

            A_soft, A_hard = generate_prediction(graph.logits.data,edge_index)
            auc, acc, pre = cal_accuracy(A, A_soft, A_hard, num_edges, epoch)            
        
            if loss_va < best_val_loss:
                best_train_loss =loss_tr
                best_val_loss = loss_va
                best_mes_from_va = loss_te
                best_auc_from_va = auc
                best_acc_from_va = acc

            logger.info(", ".join(
                                    [
                                        'Epoch: {:04d}'.format(epoch),
                                        'Train Loss: {:.8f}'.format(loss_tr),
                                        'Valid Loss: {:.8f}'.format(loss_va),
                                        # 'Picked AUC: {:.4f}'.format(best_auc_from_va),
                                        # 'Picked ACC: {:.4f}'.format(best_acc_from_va),
                                        'Current AUC: {:.4f}'.format(auc),
                                        'Current ACC: {:.4f}'.format(acc),
                                        'Current PRE: {:.4f}'.format(pre),
                                        'Time: {:.4f}s'.format(time.time()-t_start)
                                    ]
                                )
                        )
            torch.save({
                "graph": graph.state_dict(),
                "dynamics": dynamics.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "Train Loss": loss_tr,
                "Valid Loss": loss_va,
                "Test Loss": loss_te,
                "AUC": auc,
                "ACC": acc,
                "PRE": pre,
                "A_soft": A_soft,
                "args": args
            }, os.path.join(save_folder, 'checkpoint.pth'))
        
        if args.b_walltime:
            if (time.time() - t_begin) > 171900:
                break
    
    # import sys

    # # log_file = open('results_logs/dygr_'+args.suffix+'_'+str(args.sample_freq)+'_'+str(args.tr_num)+'_'+str(args.trajr_length)+'_'+str(args.K)+'_'+args.filter +'_'+ str(args.heads)+'_'+'{:1.0E}'.format(p)+'.txt', 'a')
    # sys.stdout = log_file
    # print('seed:{:08d}, auc:{:.4f}, acc:{:.4f}, last_acc:{:.4f}, train_loss:{:.8f}, test_loss:{:.8f}'.format(args.eed,best_auc_from_va,best_acc_from_va,auc,best_train_loss,best_mes_from_va))
    # sys.stdout = sys.__stdout__
    # log_file.close()
