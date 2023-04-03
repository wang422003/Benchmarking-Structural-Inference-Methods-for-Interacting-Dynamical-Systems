from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import numpy as np
import pandas as pd
import torch

import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score, average_precision_score, jaccard_score

from utils import *
from modules import *
from pipeline import *
from arg_parser import parse_args
from model import model_selection


if __name__ == '__main__':
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.save_folder:
        exp_counter = 0
        now = datetime.datetime.now()
        timestamp = now.isoformat()
        # os.mkdir(args.save_folder)
        if not os.path.exists(args.save_folder):
            os.mkdir(args.save_folder)
        if args.prior:
            save_folder = './{}/{}-it-prior-E{}-D{}-exp{}/'.format(args.save_folder, args.suffix, args.encoder,
                                                                 args.decoder, timestamp)
        elif args.data_path != '':
            name_str = args.data_path.split('/')[-4] + '_' + args.data_path.split('/')[-3] + '_' + \
                       args.data_path.split('/')[-1].split('_', 2)[-1].split('.')[0]
            # save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
            save_folder = './{}/iSIDG-{}-E{}-D{}-exp{}/'.format(args.save_folder, name_str, args.encoder,
                                                              args.decoder, timestamp)
        else:
            save_folder = './{}/{}-it-E{}-D{}-exp{}/'.format(args.save_folder, args.suffix, args.encoder,
                                                           args.decoder, timestamp)
        os.mkdir(save_folder)
        meta_file = os.path.join(save_folder, 'metadata.pkl')
        encoder_file = os.path.join(save_folder, 'encoder.pt')
        decoder_file = os.path.join(save_folder, 'decoder.pt')
        edges_folder = save_folder + 'edges_log/'
        os.mkdir(edges_folder)
        iterative_adja_folder = save_folder + 'it_logs/'
        os.mkdir(iterative_adja_folder)
        probs_folder = save_folder + 'probs/'
        os.mkdir(probs_folder)
        res_folder = save_folder + 'results/'
        os.mkdir(res_folder)

        log_file = os.path.join(save_folder, 'log.txt')
        log = open(log_file, 'w')

        pickle.dump({'args': args}, open(meta_file, "wb"))
    else:
        log = None
        save_folder = None
        edges_folder = None
        probs_folder = None
        iterative_adja_folder = None
        encoder_file = None
        decoder_file = None

        print("WARNING: No save_folder provided!" +
              "Testing (within this script) will throw an error.")

    # original:
    # train_loader, valid_loader, test_loader = load_data(args)

    train_loader, valid_loader, test_loader = load_data_benchmark(args)

    adj, rel_rec, rel_send = initialization(args)

    encoder, decoder = model_selection(args)

    if args.load_folder:
        encoder_file = os.path.join(args.load_folder, 'encoder.pt')
        encoder.load_state_dict(torch.load(encoder_file))
        decoder_file = os.path.join(args.load_folder, 'decoder.pt')
        decoder.load_state_dict(torch.load(decoder_file))

        args.save_folder = False

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=args.lr)

    if args.step_LR:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            verbose=True
        )

    # Linear indices of an upper triangular mx, used for acc calculation
    triu_indices = get_triu_offdiag_indices(args.num_nodes)
    tril_indices = get_tril_offdiag_indices(args.num_nodes)

    if args.prior:
        prior = np.array([0.98, 0.02])
        print("Using prior")
        print(prior)
        log_prior = torch.FloatTensor(np.log(prior))
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = Variable(log_prior)

        if args.cuda:
            log_prior = log_prior.cuda()
    else:
        log_prior = None

    if args.cuda:
        encoder.cuda()
        decoder.cuda()
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()
        adj = adj.cuda()
        triu_indices = triu_indices.cuda()
        tril_indices = tril_indices.cuda()

    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)

    # Train model
    t_total = time.time()
    best_val_loss = np.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        val_loss, rel_rec, rel_send, adj = train(
            epoch=epoch,
            args=args,
            best_val_loss=best_val_loss,
            rel_rec_=rel_rec,
            rel_send_=rel_send,
            adj=adj,
            log=log,
            encoder=encoder,
            decoder=decoder,
            scheduler=scheduler,
            optimizer=optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            log_prior=log_prior,
            edges_folder=edges_folder,
            probs_folder=probs_folder,
            iterative_adja_folder=iterative_adja_folder,
            encoder_file=encoder_file,
            decoder_file=decoder_file
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            print("Best val loss updated.")

    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch))
    if args.save_folder:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()

    print("------- Start testing! -------")
    test(
        args=args,
        rel_rec_=rel_rec,
        rel_send_=rel_send,
        adj=adj,
        encoder=encoder,
        decoder=decoder,
        test_loader=test_loader,
        save_folder=save_folder,
        log=log,
        encoder_file=encoder_file,
        decoder_file=decoder_file,
        structure_inference=args.only_inference
    )
    if log is not None:
        print(save_folder)
        log.close()
