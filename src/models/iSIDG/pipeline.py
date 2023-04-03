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


def train(epoch, args, best_val_loss, rel_rec_, rel_send_, adj, log, encoder, decoder, scheduler, optimizer,
          train_loader, valid_loader, log_prior, edges_folder, probs_folder, iterative_adja_folder,
          encoder_file, decoder_file):
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []
    de_train = []
    dl_train = []
    sl_train = []
    probs_train = []
    iip_it_count = 1

    prev_edge = torch.ones(args.num_nodes * args.num_nodes)
    new_edge = torch.ones(args.num_nodes * args.num_nodes)
    if args.cuda:
        prev_edge = prev_edge.cuda()
        new_edge = new_edge.cuda()

    edges_rec = np.ones((args.num_nodes, args.num_nodes))
    edges_rec = edges_rec[np.newaxis, :, :]

    encoder.train()
    decoder.train()
    if args.step_LR:
        scheduler.step()
    while iip_it_count == 1 or not iip_stop_condition(prev_edge, new_edge, args.num_nodes,
                                                      args.iip_compare_value, iip_it_count):
        for batch_idx, (data, relations) in enumerate(train_loader):
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()
            data, relations = Variable(data), Variable(relations)

            optimizer.zero_grad()

            logits = encoder(data, rel_rec_, rel_send_, adj)
            edges = gumbel_softmax(
                logits,
                tau=args.temp,
                hard=args.hard,
                multi_sample=args.multiple_sampling,
                rounds=args.multiple_sampling_rounds
            )
            prob = my_softmax(logits, -1)

            if args.store_edges:
                edges_rec_sub = inter_check(logits)
                if args.cuda:
                    edges_rec_sub = edges_rec_sub.cpu().numpy()
                edges_rec = np.concatenate((edges_rec, edges_rec_sub))

            if args.decoder == 'rnn':
                output = decoder(data, edges, rel_rec_, rel_send_, 100,
                                 burn_in=True,
                                 burn_in_steps=args.timesteps - args.prediction_steps)
            else:
                output = decoder(data, edges, rel_rec_, rel_send_,
                                 args.prediction_steps)

            target = data[:, :, 1:, :]

            loss_nll = nll_gaussian(output, target, args.var)
            if args.negative_nll_gaussian:
                loss_nll = -1 * loss_nll

            if args.prior:
                loss_kl = kl_categorical(prob, log_prior, args.num_nodes)
            else:
                loss_kl = kl_categorical_uniform(prob, args.num_nodes,
                                                 args.edge_types)

            loss_de = dirichlet_energy(adj=prob, data=data, num_nodes=args.num_nodes, cuda=args.cuda)
            # print("loss_de: {}".format(loss_de))
            # print(loss_de.size())

            loss_dl = degree_loss(adj=prob, num_nodes=args.num_nodes, cuda=args.cuda)
            # print("loss_dl: {}".format(loss_dl))
            # print(loss_dl.size())

            loss_sl = sparsity_loss(adj=prob, num_nodes=args.num_nodes)
            # print("loss_sl: {}".format(loss_sl))
            # print(loss_sl.size())

            loss = loss_nll + args.KL_weight * loss_kl + args.smoothness_weight * loss_de - \
                   args.degree_weight * loss_dl + args.sparsity_weight * loss_sl

            acc = edge_accuracy(logits, relations)
            acc_train.append(acc)
            probs_train.append(prob.detach().cpu()[:, :, 1])
            loss.backward()
            optimizer.step()

            mse_train.append(F.mse_loss(output, target).item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())
            de_train.append(loss_de.item())
            dl_train.append(loss_dl.item())
            sl_train.append(loss_sl.item())

        if args.store_edges:
            edges_save_file = edges_folder + 'edges_' + str(epoch) + '.npy'
            np.save(edges_save_file, edges_rec)
        # save probs
        if args.save_probs:
            np_probs = np.concatenate([element.numpy() for element in probs_train])
            probs_save_file = probs_folder + 'probs_' + str(epoch) + '.npy'
            np.save(probs_save_file, np_probs)
        nll_val = []
        acc_val = []
        kl_val = []
        mse_val = []
        de_val = []
        dl_val = []
        sl_val = []
        loss_val = []
        lossx_val = []
        encoder.eval()
        decoder.eval()
        for batch_idx, (data, relations) in enumerate(valid_loader):
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()
            data, relations = Variable(data, volatile=True), Variable(
                relations, volatile=True)

            logits = encoder(data, rel_rec_, rel_send_, adj)
            edges = gumbel_softmax(logits, tau=args.temp, hard=True)
            prob = my_softmax(logits, -1)

            output = decoder(data, edges, rel_rec_, rel_send_, 1)

            target = data[:, :, 1:, :]
            loss_nll = nll_gaussian(output, target, args.var)
            if args.negative_nll_gaussian:
                loss_nll = -1 * loss_nll
            loss_kl = kl_categorical_uniform(prob, args.num_nodes, args.edge_types)
            loss_de = dirichlet_energy(adj=prob, data=data, num_nodes=args.num_nodes, cuda=args.cuda)

            loss_dl = degree_loss(adj=prob, num_nodes=args.num_nodes, cuda=args.cuda)

            loss_sl = sparsity_loss(adj=prob, num_nodes=args.num_nodes)

            loss = loss_nll + args.KL_weight * loss_kl + args.smoothness_weight * loss_de - \
                   args.degree_weight * loss_dl + args.sparsity_weight * loss_sl

            loss_x = args.KL_weight * loss_kl + args.smoothness_weight * loss_de - \
                   args.degree_weight * loss_dl + args.sparsity_weight * loss_sl


            acc = edge_accuracy(logits, relations)
            acc_val.append(acc)

            loss_val.append(loss.item())
            lossx_val.append(loss_x.item())
            mse_val.append(F.mse_loss(output, target).item())
            nll_val.append(loss_nll.item())
            kl_val.append(loss_kl.item())
            de_val.append(loss_de.item())
            dl_val.append(loss_dl.item())
            sl_val.append(loss_sl.item())

        if iip_it_count >= 2:
            print("This is round {} of the present adj-m".format(iip_it_count))
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'de_train: {:.10f}'.format(np.mean(de_train)),
              'dl_train: {:.10f}'.format(np.mean(dl_train)),
              'sl_train: {:.10f}'.format(np.mean(sl_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'nll_val: {:.10f}'.format(np.mean(nll_val)),
              'kl_val: {:.10f}'.format(np.mean(kl_val)),
              'mse_val: {:.10f}'.format(np.mean(mse_val)),
              'de_val: {:.10f}'.format(np.mean(de_val)),
              'dl_val: {:.10f}'.format(np.mean(dl_val)),
              'sl_val: {:.10f}'.format(np.mean(sl_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'time: {:.4f}s'.format(time.time() - t))
        if not args.step_LR:
            if args.step_metric == 'acc':
                scheduler.step(np.mean(acc_val))
            else:
                scheduler.step(np.mean(nll_val) + np.mean(kl_val))

        if epoch < args.rounds_to_change_adj:
            print("Save the model as backup")
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)

        if args.save_folder and epoch >= args.rounds_to_change_adj and np.mean(nll_val) < best_val_loss:
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            print('Best model so far, saving...')
            if iip_it_count >= 2:
                print("This is round {} of the present adj-m".format(iip_it_count), file=log)
            print('Epoch: {:04d}'.format(epoch),
                  'nll_train: {:.10f}'.format(np.mean(nll_train)),
                  'kl_train: {:.10f}'.format(np.mean(kl_train)),
                  'mse_train: {:.10f}'.format(np.mean(mse_train)),
                  'de_train: {:.10f}'.format(np.mean(de_train)),
                  'dl_train: {:.10f}'.format(np.mean(dl_train)),
                  'sl_train: {:.10f}'.format(np.mean(sl_train)),
                  'acc_train: {:.10f}'.format(np.mean(acc_train)),
                  'nll_val: {:.10f}'.format(np.mean(nll_val)),
                  'kl_val: {:.10f}'.format(np.mean(kl_val)),
                  'mse_val: {:.10f}'.format(np.mean(mse_val)),
                  'de_val: {:.10f}'.format(np.mean(de_val)),
                  'dl_val: {:.10f}'.format(np.mean(dl_val)),
                  'sl_val: {:.10f}'.format(np.mean(sl_val)),
                  'acc_val: {:.10f}'.format(np.mean(acc_val)),
                  'loss_val: {:.10f}'.format(np.mean(loss_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            log.flush()

        iip_it_count += 1
        if epoch < args.rounds_to_change_adj - 1:
            break
        edges_sum = torch.cat(probs_train)
        prev_edge = new_edge
        new_edge = torch.sum(edges_sum, dim=0) / edges_sum.size()[0]

    # time to change the adjacency matrix:
    if epoch >= args.rounds_to_change_adj - 1:
        if best_val_loss > np.mean(loss_val):
            probs_train = torch.cat(probs_train)
            new_adj = torch.sum(probs_train, dim=0) / probs_train.size()[0]
            new_adj = new_adj.numpy()
            # Thresholding:
            new_adj = adj_thresholding(new_adj, args.avg_th_value)
            np.save(iterative_adja_folder + 'iterative{}.npy'.format(epoch), new_adj)

            if args.add_randomness:
                new_adj = modify_adja_it(new_adj, num_nodes=args.num_nodes, num_random=args.random_n)

            if args.adj_combine:
                if args.adj_norm:
                    norm_diag = adj_normalize(adj.cpu().numpy(), args.num_nodes)  # normalize the initial fully connected graph
                    if not args.adj_separate_norm:
                        new_adj_ = adj_it_normalize(new_adj, args.num_nodes)
                        new_adj_col = 0
                    else:
                        new_adj_ = adj_row_normalize(new_adj, args.num_nodes)  # in this case, for rel_rec
                        new_adj_col = adj_col_normalize(new_adj, args.num_nodes)
                else:
                    norm_diag = adj.cpu().numpy()
                    new_adj_ = new_adj
                    new_adj_col = 0
                new_adj = args.adj_combine_coe * norm_diag + (1 - args.adj_combine_coe) * new_adj_
                if args.adj_separate_norm:
                    new_adj_send = args.adj_combine_coe * norm_diag + (1 - args.adj_combine_coe) * new_adj_col
                else:
                    new_adj_send = 0
                new_adj = adj_thresholding(new_adj, args.avg_th_value)
                new_adj = np.reshape(new_adj, (args.num_nodes, args.num_nodes))
                rel_rec_ = np.array(
                    encode_onehot_change(np.reshape(new_adj, (args.num_nodes, args.num_nodes)),
                                         num_nodes=args.num_nodes, dim=0),
                    dtype=np.float32
                )
                if not args.adj_separate_norm:
                    rel_send_ = np.array(
                        encode_onehot_change(np.reshape(new_adj, (args.num_nodes, args.num_nodes)),
                                             num_nodes=args.num_nodes, dim=1),
                        dtype=np.float32
                    )
                else:
                    new_adj_send = adj_thresholding(new_adj_send, args.avg_th_value)
                    rel_send_ = np.array(
                        encode_onehot_change(np.reshape(new_adj_send, (args.num_nodes, args.num_nodes)),
                                             num_nodes=args.num_nodes, dim=1),
                        dtype=np.float32
                    )
            adj = torch.FloatTensor(new_adj)
            rel_rec_ = torch.FloatTensor(rel_rec_)
            rel_send_ = torch.FloatTensor(rel_send_)
            if args.cuda:
                rel_send_ = rel_send_.cuda()
                rel_rec_ = rel_rec_.cuda()
                adj = adj.cuda()
            print("Adjacency matrix changed!")
            print("Adjacency matrix changed!", file=log)
        else:
            print("Skip this round of iterative")
            print("Skip this round of iterative", file=log)

    return np.mean(loss_val), rel_rec_, rel_send_, adj


def test(args, rel_rec_, rel_send_, adj, encoder, decoder, test_loader,
         save_folder, log,
         encoder_file, decoder_file, structure_inference=False, ):
    acc_test = []

    nll_test = []
    kl_test = []
    mse_test = []
    de_test = []
    dl_test = []
    sl_test = []
    probs_test = []
    tot_mse = 0
    counter = 0

    edges_test = []
    logits_test = []

    auroc_test = []
    auprc_test = []
    jaccard_test = []

    encoder.eval()

    decoder.eval()
    encoder.load_state_dict(torch.load(encoder_file))
    decoder.load_state_dict(torch.load(decoder_file))

    for batch_idx, (data, relations) in enumerate(test_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data, volatile=True), Variable(
            relations, volatile=True)

        if args.timesteps <= 49 and args.suffix[:3] != 'Net':
            assert (data.size(2) - args.timesteps) >= args.timesteps

        data_encoder = data[:, :, :args.timesteps, :].contiguous()
        # old: data_decoder = data[:, :, -args.timesteps:, :].contiguous()
        data_decoder = data[:, :, :args.timesteps, :].contiguous()

        logits = encoder(data_encoder, rel_rec_, rel_send_, adj)
        edges = gumbel_softmax(logits, tau=args.temp, hard=True)

        prob = my_softmax(logits, -1)

        output = decoder(data_decoder, edges, rel_rec_, rel_send_, 1)

        target = data_decoder[:, :, 1:, :]
        loss_nll = nll_gaussian(output, target, args.var)

        if args.negative_nll_gaussian:
            loss_nll = -1 * loss_nll
        loss_kl = kl_categorical_uniform(prob, args.num_nodes, args.edge_types)

        loss_de = dirichlet_energy(adj=prob, data=data, num_nodes=args.num_nodes, cuda=args.cuda)

        loss_dl = degree_loss(adj=prob, num_nodes=args.num_nodes, cuda=args.cuda)

        loss_sl = sparsity_loss(adj=prob, num_nodes=args.num_nodes)

        if args.test_TH:
            prob = adj_thresholding(prob, args.avg_th_value, keep_origin=True)

        acc = edge_accuracy(logits, relations)
        acc_test.append(acc)

        mse_test.append(F.mse_loss(output, target).item())
        nll_test.append(loss_nll.item())
        kl_test.append(loss_kl.item())
        de_test.append(loss_de.item())
        dl_test.append(loss_dl.item())
        sl_test.append(loss_sl.item())

        logits_np = logits.detach().cpu().numpy()
        prob_np = prob.detach().cpu().numpy()
        probs_test.append(prob_np)
        edges_np = edges.detach().cpu().numpy()

        relations_np = relations.detach().cpu().numpy()
        x = torch.Tensor(prob_np)
        values, preds = x.max(-1)

        label_edges = preds.cpu().numpy()
        preds_np = prob_np[:, :, 1]
        for i in range(len(prob_np)):
            auroc = roc_auc_score(relations_np[i], preds_np[i], average=None)
            auprc = average_precision_score(relations_np[i], preds_np[i])
            if args.edge_types != 2:
                jac = jaccard_score(relations_np[i], label_edges[i], average='micro')
            else:
                jac = jaccard_score(relations_np[i], label_edges[i])

            auroc_test.append(auroc)
            auprc_test.append(auprc)
            jaccard_test.append(jac)

        logits_test.append(logits_np)
        edges_test.append(edges_np)

        if not structure_inference:
            mse = ((target - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
            tot_mse += mse.data.cpu().numpy()
        else:
            tot_mse = np.zeros(2)
        counter += 1

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')

    if not structure_inference:
        mean_mse = tot_mse / counter
        mse_str = '['
        for mse_step in mean_mse[:-1]:
            mse_str += " {:.12f} ,".format(mse_step)
        mse_str += " {:.12f} ".format(mean_mse[-1])
        mse_str += ']'
        print('nll_test: {:.10f}'.format(np.mean(nll_test)),
              'kl_test: {:.10f}'.format(np.mean(kl_test)),
              'mse_test: {:.10f}'.format(np.mean(mse_test)),
              'de_test: {:.10f}'.format(np.mean(de_test)),
              'dl_test: {:.10f}'.format(np.mean(dl_test)),
              'sl_test: {:.10f}'.format(np.mean(sl_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)))
        print('MSE: {}'.format(mse_str))

    edges_test = np.concatenate(edges_test)
    logits_test = np.concatenate(logits_test)
    np.save(save_folder + 'results/edges_test.npy', np.concatenate(probs_test))
    print("edges_test saved at: " + save_folder + 'results/edges_test.npy')

    auroc_res = np.mean(auroc_test)
    auprc_res = np.mean(auprc_test)
    jac_res = np.mean(jaccard_test)
    res = {'AUROC': auroc_res, 'AUPRC': auprc_res, 'Jaccard': jac_res}
    df = pd.DataFrame(res, index=[0])
    df.to_csv(save_folder + 'test_metrics.csv', index=False)
    print('AUROC: {}, AUPRC: {}, Jaccard: {} '.format(
        res['AUROC'], res['AUPRC'], res['Jaccard']))

    # print('Edges, logits and evaluation metrics saved at' + save_folder)

    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        if not structure_inference:
            print(
                'nll_test: {:.10f}'.format(np.mean(nll_test)),
                'kl_test: {:.10f}'.format(np.mean(kl_test)),
                'de_test: {:.10f}'.format(np.mean(de_test)),
                'dl_test: {:.10f}'.format(np.mean(dl_test)),
                'sl_test: {:.10f}'.format(np.mean(sl_test)),
                'mse_test: {:.10f}'.format(np.mean(mse_test)),
                'acc_test: {:.10f}'.format(np.mean(acc_test)),
                file=log
            )
            print('MSE: {}'.format(mse_str), file=log)
        print('AUROC: {}, AUPRC: {}, Jaccard: {}'.format(res['AUROC'], res['AUPRC'], res['Jaccard']), file=log)
        log.flush()

    print("Finished.")
    print("Dataset: ", args.suffix)
    print("Ground truth graph locates at: ", args.data_path)
    print("With portion: ", args.b_portion)
    print("With ", args.b_time_steps, " time steps")