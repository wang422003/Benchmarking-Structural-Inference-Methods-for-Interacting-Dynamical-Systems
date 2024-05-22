from __future__ import division
from __future__ import print_function

import copy
import time
import argparse
import pickle
import os
import datetime
import logging
import optuna
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score, average_precision_score, jaccard_score

from utils import *
from modules import *
from pipeline import *
from arg_parser import parse_args
from model import model_selection

LOGGING_LEVEL = logging.INFO

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
        if os.name == 'nt':
            timestamp = timestamp.replace(":", "-")
        if args.prior:
            save_folder = '{}/{}-it-prior-E{}-D{}-exp{}/'.format(args.save_folder, args.suffix, args.encoder,
                                                                 args.decoder, timestamp)
        else:
            save_folder = '{}/{}-it-E{}-D{}-exp{}/'.format(args.save_folder, args.suffix, args.encoder,
                                                           args.decoder, timestamp)
        os.mkdir(save_folder)
        meta_file = os.path.join(save_folder, 'metadata.pkl')
        encoder_file = os.path.join(save_folder, 'encoder.pt')
        decoder_file = os.path.join(save_folder, 'decoder.pt')
        rcnet_file = os.path.join(save_folder, 'rcnet.pt')
        edges_folder = save_folder + 'edges_log/'
        os.mkdir(edges_folder)
        iterative_adja_folder = save_folder + 'it_logs/'
        os.mkdir(iterative_adja_folder)
        probs_folder = save_folder + 'probs/'
        os.mkdir(probs_folder)
        ckpt_file = save_folder + 'checkpoint/'
        os.mkdir(ckpt_file)

        log_file = os.path.join(save_folder, 'log.txt')
        # log = open(log_file, 'w')
        logging.basicConfig(filename = log_file, 
                            filemode="a", 
                            format = '%(asctime)s %(name)s %(levelname)s:%(message)s',
                            level = LOGGING_LEVEL,
                            )
        log = logging.getLogger()
        tf_board_writer = SummaryWriter(log_dir = save_folder)
    else:
        save_folder = None
        edges_folder = None
        probs_folder = None
        iterative_adja_folder = None
        encoder_file = None
        decoder_file = None
        rcnet_file = None
        ckpt_file = None

        logging.basicConfig(format = '%(asctime)s %(name)s %(levelname)s:%(message)s',
                            level = LOGGING_LEVEL,
                            encoding = "utf-8"
                            )
        log = logging.getLogger()
        tf_board_writer = None

        log.warning("No save_folder provided! Testing (within this script) will throw an error.")

    log.info(args)

    train_loader, valid_loader, test_loader = load_data(args)

    adj, rel_rec, rel_send = initialization(args)

    encoder, decoder, rcnet = model_selection(args)

    current_epoch = 0
    if args.load_folder:
        encoder_file = os.path.join(args.load_folder, 'encoder.pt')
        encoder.load_state_dict(torch.load(encoder_file))
        decoder_file = os.path.join(args.load_folder, 'decoder.pt')
        decoder.load_state_dict(torch.load(decoder_file))
        rcnet_file = os.path.join(args.load_folder, 'rcnet.pt')
        rcnet.load_state_dict(torch.load(rcnet_file))

        if args.save_folder:
            encoder_file = os.path.join(save_folder, 'encoder.pt')
            decoder_file = os.path.join(save_folder, 'decoder.pt')
            rcnet_file = os.path.join(save_folder, 'rcnet.pt')

        current_epoch = max(map(int, re.findall("Epoch: (\d+)", open(os.path.join(args.load_folder, "log.txt"), "r").read())))
        current_epoch = current_epoch + 1

    # Linear indices of an upper triangular mx, used for acc calculation
    triu_indices = get_triu_offdiag_indices(args.num_nodes)
    tril_indices = get_tril_offdiag_indices(args.num_nodes)

    if args.prior:
        # prior = np.array([0.98, 0.02])
        prior = [0.01 for i in range(args.edge_types - 1)]
        prior = np.array([1-sum(prior)] + prior)
        
        log.info("Using prior")
        log.info(prior)
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
        rcnet.cuda()
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()
        adj = adj.cuda()
        triu_indices = triu_indices.cuda()
        tril_indices = tril_indices.cuda()

    if args.bome:
        xhat_encoder = copy.deepcopy(encoder)
        xhat_decoder = copy.deepcopy(decoder)
        optimizer_xhat = optim.Adam(list(xhat_encoder.parameters()) + list(xhat_decoder.parameters()),
                                    lr=args.lr)
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=args.lr)
    else:
        xhat_encoder = None
        xhat_decoder = None
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=args.lr)
        optimizer_xhat = None

    optimizer_rc = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()) + list(rcnet.parameters()), lr=args.lr)

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
    scheduler_rc = lr_scheduler.ReduceLROnPlateau(
            optimizer_rc,
            mode='min',
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            verbose=True
        )

    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)

    # Train model
    t_total = time.time()
    best_train_loss = np.inf
    best_epoch = 0
    
    # tune hyperparameters
    train_loss_history = []
    start_time = time.time()

    for epoch in range(current_epoch, args.epochs):
        if args.bome:
            train_loss, rel_rec, rel_send, adj = train(
                epoch=epoch,
                args=args,
                best_train_loss=best_train_loss,
                rel_rec_=rel_rec,
                rel_send_=rel_send,
                adj=adj,
                log=log,
                tf_board_writer=tf_board_writer,
                encoder=encoder,
                decoder=decoder,
                rcnet=rcnet,
                encoder_xhat=xhat_encoder,
                decoder_xhat=xhat_decoder,
                scheduler=scheduler,
                scheduler_rc=scheduler_rc,
                optimizer=optimizer,
                optimizer_rc=optimizer_rc,
                optimizer_inner=optimizer_xhat,
                train_loader=train_loader,
                log_prior=log_prior,
                edges_folder=edges_folder,
                probs_folder=probs_folder,
                iterative_adja_folder=iterative_adja_folder,
                encoder_file=encoder_file,
                decoder_file=decoder_file,
                rcnet_file=rcnet_file,
                is_bi_level=True
            )
        else:
            train_loss, rel_rec, rel_send, adj = train(
                epoch=epoch,
                args=args,
                best_train_loss=best_train_loss,
                rel_rec_=rel_rec,
                rel_send_=rel_send,
                adj=adj,
                log=log,
                tf_board_writer=tf_board_writer,
                encoder=encoder,
                decoder=decoder,
                rcnet=rcnet,
                scheduler=scheduler,
                scheduler_rc=scheduler_rc,
                optimizer=optimizer,
                optimizer_rc=optimizer_rc,
                train_loader=train_loader,
                log_prior=log_prior,
                edges_folder=edges_folder,
                probs_folder=probs_folder,
                iterative_adja_folder=iterative_adja_folder,
                encoder_file=encoder_file,
                decoder_file=decoder_file,
                rcnet_file=rcnet_file,
                is_bi_level=False
            )

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_epoch = epoch
            log.info("Best train loss updated.")

        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'rcnet': rcnet.state_dict(),
            "rel_rec": rel_rec,
            "rel_send": rel_send,
            "adj": adj,
            "scheduler": scheduler.state_dict(),
            "scheduler_rc": scheduler_rc.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_rc": optimizer_rc.state_dict(),
        }, os.path.join(ckpt_file, f"ckpt_{epoch}.pt"))
        
        # tune hyperparameters
        train_loss_history.append(train_loss)

        # if len(train_loss_history) > 50 and (np.array(train_loss_history)[-50:-20].max() < np.array(train_loss_history)[-20:].mean() or time.time()-start_time > 1800):
        #     tf_board_writer.close()
        #     raise optuna.TrialPruned()

    log.info("Optimization Finished!")
    log.info("Best Epoch: {:04d}".format(best_epoch))
    if args.save_folder:
        log.info("Best Epoch: {:04d}".format(best_epoch))
    tf_board_writer.close()
    # print("------- Start testing! -------")
    # test(
    #     args=args,
    #     rel_rec_=rel_rec,
    #     rel_send_=rel_send,
    #     adj=adj,
    #     encoder=encoder,
    #     decoder=decoder,
    #     rcnet=rcnet,
    #     test_loader=test_loader,
    #     save_folder=save_folder,
    #     log=log,
    #     encoder_file=encoder_file,
    #     decoder_file=decoder_file,
    #     rcnet_file=rcnet_file,
    #     structure_inference=args.only_inference
    # )
