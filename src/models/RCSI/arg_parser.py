import argparse
import torch
import datetime
import numpy as np
import logging

log = logging.getLogger()

def num_nodes(data_suffix='LI'):
    if data_suffix == 'LI':
        return 7
    elif data_suffix == 'LL':
        return 18
    elif data_suffix == 'CY':
        return 6
    elif data_suffix == 'BF':
        return 7
    elif data_suffix == 'TF':
        return 8
    elif data_suffix == 'BF-CV':
        return 10
    elif data_suffix == 'springs':
        return 5
    elif data_suffix == 'charged':
        return 5
    elif data_suffix == 'kuramoto':
        return 5
    elif data_suffix == 'NetSim1':
        return 5
    elif data_suffix == 'NetSim2':
        return 10
    elif data_suffix == 'NetSim3':
        return 15
    elif data_suffix == 'scRNAseq':
        return 25
    else:
        raise ValueError("Check the suffix of the dataset!")


def num_dims(data_suffix='LI'):
    if data_suffix == 'LI':
        return 1
    elif data_suffix == 'LL':
        return 1
    elif data_suffix == 'CY':
        return 1
    elif data_suffix == 'BF':
        return 1
    elif data_suffix == 'TF':
        return 1
    elif data_suffix == 'BF-CV':
        return 1
    elif data_suffix == 'springs':
        return 4
    elif data_suffix == 'charged':
        return 4
    elif data_suffix == 'kuramoto':
        return 4
    elif data_suffix == 'NetSim1':
        return 1
    elif data_suffix == 'NetSim2':
        return 1
    elif data_suffix == 'NetSim3':
        return 1
    elif data_suffix == 'scRNAseq':
        return 1
    else:
        raise ValueError("Check the suffix of the dataset!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Number of samples per batch.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate.')
    parser.add_argument('--encoder-hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--decoder-hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='Temperature for Gumbel softmax.')
    # parser.add_argument('--num-nodes', type=int, default=7,
    #                     help='Number of nodes in network.')
    # # CY: 6, LI: 7, LL: 18
    parser.add_argument('--encoder', type=str, default='gin',
                        help='Type of path encoder model (mlp, mpm, gin, gcn, or gat).')
    parser.add_argument('--decoder', type=str, default='mlp',
                        help='Type of decoder model (mlp, multi, or rnn).')
    parser.add_argument('--no-factor', action='store_true', default=False,
                        help='Disables factor graph model.')
    parser.add_argument('--suffix', type=str, default='LI',
                        help='Suffix for training data (e.g. "LI", "LL", "CY", "BF", "TF", "BF-CV", "springs", "charged", "kuramoto", "NetSim1", "NetSim2", "NetSim3", "scRNAseq").')
    parser.add_argument('--encoder-dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--decoder-dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--save-folder', type=str, default='./logs',
                        help='Where to save the trained model, leave empty to not save anything.')
    parser.add_argument('--load-folder', type=str, default='',
                        help='Where to load the trained model if finetunning. ' +
                             'Leave empty to train from scratch')
    parser.add_argument('--edge-types', type=int, default=2,
                        help='The number of edge types to infer.')
    # parser.add_argument('--dims', type=int, default=1,
    #                     help='The number of input dimensions per trajectory only one.')
    parser.add_argument('--timesteps', type=int, default=49,
                        help='The number of time steps per sample.')
    # parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
    #                     help='Num steps to predict before re-using teacher forcing.')
    parser.add_argument('--lr-decay', type=int, default=100,
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR decay factor.')
    parser.add_argument('--skip-first', action='store_true', default=False,
                        help='Skip first edge type in decoder, i.e. it represents no-edge.')
    parser.add_argument('--var', type=float, default=5e-5,
                        help='Output variance.')
    parser.add_argument('--hard', action='store_true', default=False,
                        help='Uses discrete samples in training forward pass.')
    parser.add_argument('--prior', action='store_true', default=False,
                        help='Whether to use sparsity prior.')
    parser.add_argument('--include-self-loops', action='store_true', default=False,
                        help='Whether include self loops during training')
    parser.add_argument('--only-inference', action='store_true', default=False,
                        help='Whether the model is only supposed to do the inference')
    parser.add_argument('--step-learning', action='store_true', default=False,
                        help='Whether to train the model firstly with supervised then with unsupervised.')
    parser.add_argument('--store-edges', action='store_true', default=False,
                        help='Whether to store the output of encoder during the training process.')
    parser.add_argument('--step-LR', action='store_true', default=False,
                        help='Whether to use Step-LR or ReduceLROnPlateau. If "True", '
                             'the optimizer with be scheduled by step_LR, otherwise, Reduce Learning-rate on Plateau.')
    parser.add_argument('--plateau-factor', type=float, default=0.5, help='The factor for ReduceLROnPlateau.')
    parser.add_argument('--plateau-patience', type=int, default=50, help='The patience for ReduceLROnPlateau.')
    parser.add_argument('--step-metric', type=str, default='loss',
                        help='Metric for ReduceLROnPlateau, otherwise, it will choose loss.')
    parser.add_argument('--negative-nll-gaussian', action='store_true', default=False,
                        help='Whether to change the nll_gaussian loss to negative value '
                             '(in the original implementation is positive).')
    parser.add_argument('--multiple-sampling', action='store_true', default=False,
                        help='Whether to sample multiple times from "Gumbel Softmax".')
    parser.add_argument('--multiple-sampling-rounds', type=int, default=3,
                        help='The number of sampling rounds from "Gumbel Softmax".')

    # args specially designed for iterative training:
    parser.add_argument('--rounds-to-change-adj', type=int, default=150,
                        help='After how many rounds to substitute the fully graph structure in the encoder.')
    parser.add_argument('--add-randomness', action='store_true', default=False,
                        help='Add randomness to the adjacency matrix the one to encoder.')

    parser.add_argument('--avg-th-value', type=float, default=0.15,
                        help='The threshold value for the adjacency matrix for iterative process.')
    parser.add_argument('--adj-norm', action='store_true', default=False,
                        help='Normalize the adjacency matrix before the iteration.')
    parser.add_argument('--adj-separate-norm', action='store_true', default=False,
                        help='Normalize the adjacency matrix before the iteration in row-wise for rel_rec, '
                             'and column-wise for rel_send.')
    parser.add_argument('--adj-combine', action='store_true', default=False,
                        help='Sum the adjacency matrix of the last round and the initial full graph.')
    parser.add_argument('--adj-combine-coe', type=float, default=0.3,
                        help='The weight for a the combination of adjacency matrix of initial and the previous round.')
    parser.add_argument('--random-n', type=int, default=4,
                        help='The number of random connections to add in the adjacency matrix.')
    parser.add_argument('--test-TH', action='store_true', default=False,
                        help='Also use threshold in the test.')

    # For loss :
    parser.add_argument('--KL-weight', type=float, default=200, help='The weight for KL divergence, default = 200.0.')
    parser.add_argument('--smoothness-weight', type=float, default=50,
                        help='The weight for Dirichlet energy, default = 50.0.')
    parser.add_argument('--degree-weight', type=float, default=10, help='The weight for degree loss, default = 10.0.')
    parser.add_argument('--sparsity-weight', type=float, default=20,
                        help='The weight for sparsity loss, default = 20.0.')

    # Inter iterative process:
    parser.add_argument('--iip-compare-value', type=float, default=0.00001,
                        help='The coefficient for iip stop condition.')
    # save probs:
    parser.add_argument('--save-probs', action='store_true', default=False,
                        help='Save the probs during training.')
    # mutual information calculation:
    parser.add_argument('--cal-mi', action='store_true', default=False,
                        help='Calculate the mutual information and save the results.')
    
    # Args for RC training:
    parser.add_argument('--rc-rounds', type=int, default=100,
                        help='Total rounds to train RC before return adj to the input side.')
    parser.add_argument('--rc-freeze-rounds', type=int, default=30,
                        help='How many rounds to perform teaching on RC.')
    parser.add_argument('--bome', action='store_true', default=False,
                        help='Whether use BOME to perform bi-level optimization.')
    parser.add_argument('--bome-inner-steps', type=int, default=10,
                        help='Total rounds to perform gradient descent on the inner optimization.')
    parser.add_argument('--bome-u1', type=float, default=0.5,
                        help='The coefficient for BOME calculation.')

    # portion training for RC
    parser.add_argument('--RC-portion', type=float, default=1.0,
                        help='Portion of data to be used in RC.')
    parser.add_argument('--RC-time-steps', type=int, default=49,
                        help='Portion of time series in data to be used in RC.')
    parser.add_argument('--RC-shuffle', action='store_true', default=False,
                        help='Shuffle the data for RC?.')
    parser.add_argument('--manual-nodes', type=int, default=0,
                        help='The number of nodes if changed from the original dataset.')
    
    # portion training for scRNAseq
    parser.add_argument("--file-name", type=str, default="./data/trajectory.npy")

    args = parser.parse_args()
    if args.suffix == 'scRNAseq':
        args.num_nodes = np.load(args.file_name).shape[-1]
    else:
        args.num_nodes = num_nodes(data_suffix=args.suffix)
        
    args.dims = num_dims(data_suffix=args.suffix)
    
    if args.decoder == 'multi':
        args.prediction_steps = 10
    else:
        args.prediction_steps = 1

    if args.encoder == 'gat':
        args.batch_size = 1

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.factor = not args.no_factor
    log.info(args)
    return args


if __name__ == "__main__":
    args_ = parse_args()
    log.info(args_)
