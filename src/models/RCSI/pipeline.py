from __future__ import division
from __future__ import print_function

import copy
import time
import numpy as np
import torch

from utils import *
from modules import *


def train(
    epoch,
    args,
    best_train_loss,
    rel_rec_,
    rel_send_,
    adj,
    log,
    tf_board_writer,
    encoder,
    decoder,
    rcnet,
    scheduler,
    scheduler_rc,
    optimizer,
    optimizer_rc,
    train_loader,
    log_prior,
    edges_folder,
    probs_folder,
    iterative_adja_folder,
    encoder_file,
    decoder_file,
    rcnet_file,
    is_bi_level=False,
    encoder_xhat=None,
    decoder_xhat=None,
    optimizer_inner=None
):
    # modified to add RC training

    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    nrmse_train = []
    de_train = []
    dl_train = []
    sl_train = []
    probs_train = []
    iip_it_count = 1
    device = torch.device("cuda" if args.cuda else "cpu")

    prev_edge = torch.ones(args.num_nodes * args.num_nodes).to(device)
    new_edge = torch.ones(args.num_nodes * args.num_nodes).to(device)

    edges_rec = np.ones((args.num_nodes, args.num_nodes))
    edges_rec = edges_rec[np.newaxis, :, :]

    encoder.train()
    decoder.train()
    rcnet.train()
    if args.step_LR:
        scheduler.step()
    while iip_it_count == 1 or not iip_stop_condition(prev_edge, new_edge, args.num_nodes,
                                                      args.iip_compare_value, iip_it_count):
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            data.requires_grad = True

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
                edges_rec_sub = inter_check(logits, args)
                edges_rec = np.concatenate((edges_rec, edges_rec_sub))

            if args.decoder == 'rnn':
                output, aug_inputs = decoder(data, edges, rel_rec_, rel_send_, 100,
                                 burn_in=True,
                                 burn_in_steps=args.timesteps - args.prediction_steps)
            else:
                output, aug_inputs = decoder(data, edges, rel_rec_, rel_send_,
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
            loss_dl = degree_loss(adj=prob, num_nodes=args.num_nodes, cuda=args.cuda)
            loss_sl = sparsity_loss(adj=prob, num_nodes=args.num_nodes)
            loss = loss_nll + args.KL_weight * loss_kl + args.smoothness_weight * loss_de - \
                   args.degree_weight * loss_dl + args.sparsity_weight * loss_sl

            probs_train.append(prob)
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
            np_probs = np.concatenate([element.detach().cpu().numpy() for element in probs_train])
            probs_save_file = probs_folder + 'probs_' + str(epoch) + '.npy'
            np.save(probs_save_file, np_probs)

        if iip_it_count >= 2:
            log.info("This is round {} of the present adj-m".format(iip_it_count))
        log.info(
            ", ".join(
                [
                    'Epoch: {:04d}'.format(epoch),
                    'nll_train: {:.10f}'.format(np.mean(nll_train)),
                    'kl_train: {:.10f}'.format(np.mean(kl_train)),
                    'mse_train: {:.10f}'.format(np.mean(mse_train)),
                    'de_train: {:.10f}'.format(np.mean(de_train)),
                    'dl_train: {:.10f}'.format(np.mean(dl_train)),
                    'sl_train: {:.10f}'.format(np.mean(sl_train)),
                    'time: {:.4f}s'.format(time.time() - t)
                ]
            )
        )

        if tf_board_writer is not None:
            if args.save_probs:
                update_tensorboard(tf_board_writer=tf_board_writer, epoch=epoch, 
                                nll_train=nll_train, kl_train=kl_train, mse_train=mse_train,
                                de_train=de_train, dl_train=dl_train, sl_train=sl_train, prob = np_probs)
            else:
                update_tensorboard(tf_board_writer=tf_board_writer, epoch=epoch, 
                                nll_train=nll_train, kl_train=kl_train, mse_train=mse_train,
                                de_train=de_train, dl_train=dl_train, sl_train=sl_train)

        if not args.step_LR:
            scheduler.step(np.mean(nll_train) + np.mean(kl_train))

        if epoch < args.rounds_to_change_adj:
            log.info("Save the model as backup")
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            torch.save(rcnet.state_dict(), rcnet_file)

        if args.save_folder and epoch >= args.rounds_to_change_adj and np.mean(nll_train) < best_train_loss:
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            torch.save(rcnet.state_dict(), rcnet_file)
            log.info('Best model so far, saving...')
            if iip_it_count >= 2:
                log.info("This is round {} of the present adj-m".format(iip_it_count))
            log.info(
                ", ".join(
                    [
                        'Epoch: {:04d}'.format(epoch),
                        'nll_train: {:.10f}'.format(np.mean(nll_train)),
                        'kl_train: {:.10f}'.format(np.mean(kl_train)),
                        'mse_train: {:.10f}'.format(np.mean(mse_train)),
                        'de_train: {:.10f}'.format(np.mean(de_train)),
                        'dl_train: {:.10f}'.format(np.mean(dl_train)),
                        'sl_train: {:.10f}'.format(np.mean(sl_train)),
                        'time: {:.4f}s'.format(time.time() - t)
                    ]
                )
            )

        iip_it_count += 1
        if epoch < args.rounds_to_change_adj - 1:
            break
        edges_sum = torch.cat([p[:,:,1:].sum(axis=-1).float() for p in probs_train])
        prev_edge = new_edge
        new_edge = torch.sum(edges_sum, dim=0) / edges_sum.size()[0]
    # --------------------------------- #
    # RC here!
    for rc_iter in range(args.rc_rounds):
        log.info("RC now~")
        n_params_w = sum(p.numel() for p in rcnet.parameters() if p.requires_grad)
        zz = torch.zeros(n_params_w).to(device)
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)

            if rc_iter < args.rc_freeze_rounds:
                for parameter in encoder.parameters():
                    parameter.requires_grad = False
                for parameter in decoder.parameters():
                    parameter.requires_grad = False

                if is_bi_level:
                    optimizer_rc.zero_grad()

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
                        edges_rec_sub = inter_check(logits, args)
                        edges_rec = np.concatenate((edges_rec, edges_rec_sub))

                    if args.decoder == 'rnn':
                        output, aug_inputs = decoder(
                            data, edges, rel_rec_, rel_send_, 100,
                            burn_in=True,
                            burn_in_steps=args.timesteps - args.prediction_steps
                        )
                    else:
                        output, aug_inputs = decoder(
                            data, edges, rel_rec_, rel_send_,
                            args.prediction_steps
                        )
                    rc_output = rcnet(aug_inputs)

                    target = data[:, :, 1:, :]

                    loss_rc = nrmse_loss(rc_output, target)

                    loss = loss_rc
                    nrmse_train.append(loss_rc.item())

                    loss.backward()
                    optimizer_rc.step()
            elif rc_iter == args.rc_freeze_rounds:
                for parameter in encoder.parameters():
                    parameter.requires_grad = True
                for parameter in decoder.parameters():
                    parameter.requires_grad = True
            else:
                continue

            data = data.to(device)
            data.requires_grad = True

            optimizer_rc.zero_grad()

            if is_bi_level:
                # This time with bi-level optimization
                # We use BOME! Many thanks author of BOME for the implementation details!
                if rc_iter >= args.rc_freeze_rounds:
                    encoder_xhat.load_state_dict(copy.deepcopy(encoder.state_dict()))
                    decoder_xhat.load_state_dict(copy.deepcopy(decoder.state_dict()))
                    for it in range(args.bome_inner_steps):

                        optimizer_inner.zero_grad()

                        logits = encoder_xhat(data, rel_rec_, rel_send_, adj)
                        edges = gumbel_softmax(
                            logits,
                            tau=args.temp,
                            hard=args.hard,
                            multi_sample=args.multiple_sampling,
                            rounds=args.multiple_sampling_rounds
                        )
                        prob = my_softmax(logits, -1)

                        if args.store_edges:
                            edges_rec_sub = inter_check(logits, args)
                            edges_rec = np.concatenate((edges_rec, edges_rec_sub))

                        if args.decoder == 'rnn':
                            output, aug_inputs = decoder_xhat(
                                data, edges, rel_rec_, rel_send_, 100,
                                burn_in=True,
                                burn_in_steps=args.timesteps - args.prediction_steps
                            )
                        else:
                            output, aug_inputs = decoder(
                                data, edges, rel_rec_, rel_send_,
                                args.prediction_steps
                            )
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
                        loss_dl = degree_loss(adj=prob, num_nodes=args.num_nodes, cuda=args.cuda)
                        loss_sl = sparsity_loss(adj=prob, num_nodes=args.num_nodes)
                        loss_xhat = loss_nll + args.KL_weight * loss_kl + args.smoothness_weight * loss_de - \
                            args.degree_weight * loss_dl + args.sparsity_weight * loss_sl
                        loss_xhat.backward()
                        optimizer_inner.step()

                    loss, gx, fx, gw_minus_gw_k, loss_rc, loss_kl, loss_de, loss_dl, loss_sl, logits, prob = f_x_n_g_x(
                        encoder=encoder,
                        decoder=decoder,
                        encoder_xhat=encoder_xhat,
                        decoder_xhat=decoder_xhat,
                        rcnet=rcnet,
                        data=data,
                        rel_rec=rel_rec_,
                        rel_send=rel_send_,
                        adj=adj,
                        args=args,
                        log_prior=log_prior
                    )
                    log.info("fx: ")
                    log.info(fx.size())
                    log.info("gx: ")
                    log.info(gx.size())
                    log.info("gw_minus_gw_k: ")
                    log.info(gw_minus_gw_k.size())
                    log.info("zz: ")
                    log.info(zz.size())
                    df = fx.view(-1)
                    dg = torch.add(gx.view(-1), gw_minus_gw_k.view(-1))
                    log.info("df: ")
                    log.info(df.size())
                    log.info("dg: ")
                    log.info(dg.size())
                    norm_dq = dg.norm().pow(2)
                    log.info("Norm_dq: ")
                    log.info(norm_dq.size())
                    dot = df.dot(dg)
                    lmbd = F.relu((args.bome_u1 * loss - dot) / (norm_dq + 1e-8))

                    optimizer_rc.zero_grad()
                    encoder.grad = fx + lmbd * gx
                    decoder.grad = fx + lmbd * gx
                    rcnet.grad = lmbd * gw_minus_gw_k
                    optimizer_rc.step()

                    target = data[:, :, 1:, :]

                    probs_train.append(prob)

                    nrmse_train.append(loss_rc.item())
                    kl_train.append(loss_kl.item())
                    de_train.append(loss_de.item())
                    dl_train.append(loss_dl.item())
                    sl_train.append(loss_sl.item())
            else:
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
                    edges_rec_sub = inter_check(logits, args)
                    edges_rec = np.concatenate((edges_rec, edges_rec_sub))

                if args.decoder == 'rnn':
                    output, aug_inputs = decoder(
                        data, edges, rel_rec_, rel_send_, 100,
                        burn_in=True,
                        burn_in_steps=args.timesteps - args.prediction_steps)
                else:
                    output, aug_inputs = decoder(
                        data, edges, rel_rec_, rel_send_,
                        args.prediction_steps
                    )

                rc_output = rcnet(aug_inputs)

                target = data[:, :, 1:, :]

                loss_rc = nrmse_loss(rc_output, target)

                if rc_iter < args.rc_freeze_rounds:
                    loss = loss_rc
                    nrmse_train.append(loss_rc.item())
                else:
                    if args.prior:
                        loss_kl = kl_categorical(prob, log_prior, args.num_nodes)
                    else:
                        loss_kl = kl_categorical_uniform(prob, args.num_nodes,
                                                        args.edge_types)

                    loss_de = dirichlet_energy(adj=prob, data=data, num_nodes=args.num_nodes, cuda=args.cuda)
                    loss_dl = degree_loss(adj=prob, num_nodes=args.num_nodes, cuda=args.cuda)
                    loss_sl = sparsity_loss(adj=prob, num_nodes=args.num_nodes)
                    loss = loss_rc + args.KL_weight * loss_kl + args.smoothness_weight * loss_de - \
                        args.degree_weight * loss_dl + args.sparsity_weight * loss_sl

                    mse_train.append(F.mse_loss(rc_output, target).item())
                    nrmse_train.append(loss_rc.item())
                    kl_train.append(loss_kl.item())
                    de_train.append(loss_de.item())
                    dl_train.append(loss_dl.item())
                    sl_train.append(loss_sl.item())

                probs_train.append(prob)
                loss.backward()
                optimizer.step()

        if args.store_edges:
            edges_save_file = edges_folder + 'edges_' + str(epoch) + 'rc.npy'
            np.save(edges_save_file, edges_rec)
        # save probs
        if args.save_probs:
            np_probs = np.concatenate([element.detach().cpu().numpy() for element in probs_train])
            probs_save_file = probs_folder + 'probs_' + str(epoch) + 'rc.npy'
            np.save(probs_save_file, np_probs)

        if iip_it_count >= 2:
            log.info("This is round {} of the present adj-m".format(iip_it_count))
        log.info(
            ", ".join(
                [
                    'Epoch: {:04d}'.format(epoch),
                    'nll_train: {:.10f}'.format(np.mean(nll_train)),
                    'kl_train: {:.10f}'.format(np.mean(kl_train)),
                    'mse_train: {:.10f}'.format(np.mean(mse_train)),
                    'nrmse_train: {:.10f}'.format(np.mean(nrmse_train)),
                    'de_train: {:.10f}'.format(np.mean(de_train)),
                    'dl_train: {:.10f}'.format(np.mean(dl_train)),
                    'sl_train: {:.10f}'.format(np.mean(sl_train)),
                    'time: {:.4f}s'.format(time.time() - t)
                ]
            )
        )

        if tf_board_writer is not None:
            if args.save_probs:
                update_tensorboard(tf_board_writer=tf_board_writer, epoch=epoch, 
                                nll_train=nll_train, kl_train=kl_train, mse_train=mse_train, nrmse_train=nrmse_train,
                                de_train=de_train, dl_train=dl_train, sl_train=sl_train, prob = np_probs)
            else:
                update_tensorboard(tf_board_writer=tf_board_writer, epoch=epoch, 
                                nll_train=nll_train, kl_train=kl_train, mse_train=mse_train, nrmse_train=nrmse_train,
                                de_train=de_train, dl_train=dl_train, sl_train=sl_train)

        if not args.step_LR:
            scheduler_rc.step(np.mean(nrmse_train) + np.mean(kl_train))

        if epoch < args.rounds_to_change_adj:
            log.info("Save the model as backup")
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            torch.save(rcnet.state_dict(), rcnet_file)

        if args.save_folder and epoch >= args.rounds_to_change_adj and np.mean(nll_train) < best_train_loss:
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            torch.save(rcnet.state_dict(), rcnet_file)
            log.info('Best model so far, saving...')
            if iip_it_count >= 2:
                log.info("This is round {} of the present adj-m".format(iip_it_count))
            log.info(
                ", ".join(
                    [
                        'Epoch: {:04d}'.format(epoch),
                        'nll_train: {:.10f}'.format(np.mean(nll_train)),
                        'kl_train: {:.10f}'.format(np.mean(kl_train)),
                        'mse_train: {:.10f}'.format(np.mean(mse_train)),
                        'nrmse_train: {:.10f}'.format(np.mean(nrmse_train)),
                        'de_train: {:.10f}'.format(np.mean(de_train)),
                        'dl_train: {:.10f}'.format(np.mean(dl_train)),
                        'sl_train: {:.10f}'.format(np.mean(sl_train)),
                        'time: {:.4f}s'.format(time.time() - t)
                    ]
                )
            )

        iip_it_count += 1
        if epoch < args.rounds_to_change_adj - 1:
            break
        edges_sum = torch.cat([p[:,:,1:].sum(axis=-1).float() for p in probs_train])
        prev_edge = new_edge
        new_edge = torch.sum(edges_sum, dim=0) / edges_sum.size()[0]

    # time to change the adjacency matrix:
    if epoch >= args.rounds_to_change_adj - 1:
        if best_train_loss > np.mean(nll_train):
            probs_train = torch.cat([p[:,:,1:].sum(axis=-1).float() for p in probs_train])
            # if probs_train.mean() < 0.5:
            #     probs_train = 1 - probs_train
                
            new_adj = torch.sum(probs_train, dim=0) / probs_train.size()[0]
            new_adj = new_adj.detach().cpu().numpy()
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
            new_adj = torch.tensor(new_adj).to(device)
            adj = new_adj
            rel_rec_ = torch.tensor(rel_rec_).to(device)
            rel_send_ = torch.tensor(rel_send_).to(device)
            log.info("Adjacency matrix changed!")
        else:
            log.info("Skip this round of iterative")

    return np.mean(nll_train), rel_rec_, rel_send_, adj


def f_x_n_g_x(encoder, decoder, encoder_xhat, decoder_xhat,
              rcnet, data, rel_rec, rel_send, adj, args, log_prior):
    """
    Calculate the delta f (gradient of outer function according to BOME)
    :param decoder_xhat:
    :param encoder_xhat:
    :param log_prior:
    :param args:
    :param adj:
    :param rel_send:
    :param rel_rec:
    :param encoder:
    :param decoder:
    :param rcnet:
    :param data:
    :return:
    """
    logits = encoder(data, rel_rec, rel_send, adj)
    edges = gumbel_softmax(
        logits,
        tau=args.temp,
        hard=args.hard,
        multi_sample=args.multiple_sampling,
        rounds=args.multiple_sampling_rounds
    )
    prob = my_softmax(logits, -1)

    if args.decoder == 'rnn':
        output, aug_inputs = decoder(data, edges, rel_rec, rel_send, 100,
                         burn_in=True,
                         burn_in_steps=args.timesteps - args.prediction_steps)
    else:
        output, aug_inputs = decoder(data, edges, rel_rec, rel_send,
                         args.prediction_steps)
    rc_output = rcnet(aug_inputs)
    loss, loss_rc, loss_kl, loss_de, loss_dl, loss_sl = total_rc_loss(
        data=data,
        output=output,
        rc_output=rc_output,
        prob=prob,
        args=args,
        log_prior=log_prior
    )
    loss.backward()

    logits_xhat = encoder_xhat(data, rel_rec, rel_send, adj)
    edges_xhat = gumbel_softmax(
        logits_xhat,
        tau=args.temp,
        hard=args.hard,
        multi_sample=args.multiple_sampling,
        rounds=args.multiple_sampling_rounds
    )
    prob_xhat = my_softmax(logits_xhat, -1)

    if args.decoder == 'rnn':
        output_xhat, aug_inputs_xhat = decoder_xhat(data, edges_xhat, rel_rec, rel_send, 100,
                                     burn_in=True,
                                     burn_in_steps=args.timesteps - args.prediction_steps)
    else:
        output_xhat, aug_inputs_xhat = decoder_xhat(data, edges_xhat, rel_rec, rel_send,
                                     args.prediction_steps)
    rc_output_xhat = rcnet(aug_inputs_xhat)
    loss_xhat, loss_rc_xhat, loss_kl_xhat, loss_de_xhat, loss_dl_xhat, loss_sl_xhat = total_rc_loss(
        data=data,
        output=output_xhat,
        rc_output=rc_output_xhat,
        prob=prob_xhat,
        args=args,
        log_prior=log_prior
    )
    loss_xhat.backward()

    return loss, output, rc_output, output - output_xhat, loss_rc, loss_kl, loss_de, loss_dl, loss_sl, logits, prob