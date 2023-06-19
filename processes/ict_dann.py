import numpy as np
from time import time
import os
import torch
from torch import nn
from torch.distributions import Beta
from itertools import cycle

from tools import save_model, print_mixup_ict_results, set_optimizer, get_model, get_data, get_transformation

__author__ = 'Shayan Gharib'
__docformat__ = 'reStructuredText'
__all__ = ['ict_dann']


def ict_dann(params, device, index):
    """the proposed method (ICT+DANN)

    :param params: settings
    :param device: device GPU or CPU
    :param index: the index of the corresponding experiment

    :return:
    """

    data_param_dict = {'batch_size': int(params['data']['batch_size'] / 2),
                       'shuffle': params['data']['shuffle'],
                       'drop_last': params['data']['drop_last'],
                       'n_workers': params['data']['nb_workers']
                       }

    source_transform, target_transform = get_transformation(source=params['training']['source_dataset_name'],
                                                            target=params['training']['target_dataset_name'])

    src_tr_data = get_data(dataset_name=params['training']['source_dataset_name'],
                           other_dataset=params['training']['target_dataset_name'].lower(),
                           transform_function=source_transform, split='train', **data_param_dict)

    src_val_data = get_data(dataset_name=params['training']['source_dataset_name'],
                            transform_function=source_transform, split='test', **data_param_dict)

    trgt_tr_data = get_data(dataset_name=params['training']['target_dataset_name'],
                            other_dataset=params['training']['source_dataset_name'].lower(),
                            transform_function=target_transform, split='train', **data_param_dict)

    epochs = params['training']['nb_epochs']
    patience = params['training']['patience']
    early_stopping = params['training']['early_stopping']

    source_type = params['training']['source_dataset_name']

    grl_multiplier = params['training']['grl_multiplier']
    mixup_beta_dist_param = params['training']['mixup']['beta_parameter']

    do_rampup = params['training']['ict']['do_rampup']
    ict_loss = params['training']['ict']['loss']
    ict_start_epoch = params['training']['ict']['ict_start_epoch']
    ict_alpha = params['training']['ict']['alpha']
    do_conf_thr = params['training']['ict']['do_confidence_threshold']
    conf_thr = params['training']['ict']['confidence_threshold']

    feature_extractor, classifier, discriminator = get_model(model_type=source_type, device=device,
                                                             target_name=params['training']['target_dataset_name'],
                                                             nb_disc_output=2)
    feature_extractor_ema, classifier_ema, _ = get_model(model_type=source_type, device=device,
                                                         target_name=params['training']['target_dataset_name'],
                                                         ema_model=True)

    optimizer = set_optimizer(opt_config=params['training']['optimizer'],
                              model_params=list(feature_extractor.parameters()) + list(classifier.parameters()) + list(
                                  discriminator.parameters())
                              )

    best_val_loss = 10000
    patience_cnt = 0
    best_epoch = 0

    current_step = 0

    nb_iter = min(len(src_tr_data), len(trgt_tr_data))
    total_iterations = epochs * nb_iter

    for ep_ in range(epochs):
        start_time = time()

        start_step = ep_ * nb_iter

        feature_extractor, classifier, discriminator, tr_cls_loss, tr_da_loss, tr_cls_acc, tr_da_acc, \
            current_step, consistency_loss = _training(src_data=src_tr_data, trgt_data=trgt_tr_data,
                                                       mdl=feature_extractor, cls=classifier,
                                                       mdl_ema=feature_extractor_ema, cls_ema=classifier_ema,
                                                       disc=discriminator, opt_=optimizer, device=device,
                                                       beta_dist_param=mixup_beta_dist_param,
                                                       grl_multiplier=grl_multiplier, step_=current_step,
                                                       do_rampup=do_rampup, current_epoch=ep_+1,
                                                       start_step=start_step, total_steps=total_iterations,
                                                       ict_loss=ict_loss, ict_start_epoch=ict_start_epoch,
                                                       ict_alpha=ict_alpha, do_conf_thr=do_conf_thr,
                                                       conf_thr=conf_thr
                                                       )

        feature_extractor, classifier, val_loss, val_acc = _validation(data=src_val_data,
                                                                       mdl=feature_extractor, cls=classifier,
                                                                       device=device
                                                                       )

        end_time = time() - start_time

        print_mixup_ict_results(ep=ep_+1, loss_tr=tr_cls_loss, loss_val=val_loss, loss_domain=tr_da_loss,
                                loss_consistency=consistency_loss, tr_acc=tr_cls_acc, val_acc=val_acc,
                                acc_domain=tr_da_acc, time_=end_time)

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt = 0

                best_epoch = ep_ + 1

                save_model(model=feature_extractor, base_dir=os.path.join(
                    params['training']['models']['base_dir_output'], index),
                           model_f_name='ict_' + params['training']['models']['model_f_name'])
                save_model(model=classifier, base_dir=os.path.join(
                    params['training']['models']['base_dir_output'], index),
                           model_f_name='ict_' + params['training']['models']['classifier_f_name'])
            else:
                patience_cnt += 1

            if patience_cnt > patience:
                print('\n  -- Epoch: {} -- Best validation loss: {}\n\n'.format(best_epoch, best_val_loss))
                break

    if not early_stopping:
        save_model(model=feature_extractor, base_dir=os.path.join(
            params['training']['models']['base_dir_output'], index),
                   model_f_name='ict_' + params['training']['models']['model_f_name'])
        save_model(model=classifier, base_dir=os.path.join(
            params['training']['models']['base_dir_output'], index),
                   model_f_name='ict_' + params['training']['models']['classifier_f_name'])

        print('\n  -- The training is done. Models (of last epoch) are saved. \n\n')
    return


def _training(src_data, trgt_data, mdl, cls, mdl_ema, cls_ema, disc, device, opt_,
              beta_dist_param, grl_multiplier, step_, do_rampup, current_epoch, start_step, total_steps,
              ict_loss, ict_start_epoch, ict_alpha, do_conf_thr, conf_thr):
    """training function for our method

    :param src_data: source training data
    :param trgt_data: target training data
    :param mdl: feature extractor student
    :param cls: classifier
    :param mdl_ema: feature extractor teacher
    :param cls_ema: classifier teacher
    :param disc: discriminator
    :param device: device GPU or CPU
    :param opt_: optimizer
    :param beta_dist_param: the parameter for Beta distribution
    :param grl_multiplier: the constant multiplier for GRL
    :param step_: current step for ema optimization
    :param do_rampup: whether to ramp up the importance of domain and ict losses
    :param current_epoch: current epoch
    :param start_step: current iteration
    :param total_steps: total number of iterations
    :param ict_loss: the type of ICT loss
    :param ict_start_epoch: the start epoch for ICT to be integrated into training process
    :param ict_alpha: ICT EMA rate
    :param do_conf_thr: whether to do confidence threshold
    :param conf_thr:  the confidence threshold rate

    :return:feature extractor, classifier, discriminator, classification loss, domain loss, classification accuracy,
    domain accuracy, current step for ema optimization, and consistency loss (ICT loss)
    """

    epoch_cls_loss = []
    epoch_da_loss = []
    epoch_cons_loss = []

    targets = []
    predictions = []

    domain_targets = []
    domain_predictions = []

    mdl.train()
    cls.train()
    mdl_ema.train()
    cls_ema.train()

    src_len = len(src_data)
    trgt_len = len(trgt_data)

    for_loop_zip = zip(src_data, cycle(trgt_data)) if src_len > trgt_len else zip(cycle(src_data), trgt_data)

    for iter_i, (src_item, trgt_item) in enumerate(for_loop_zip):
        x_src = src_item[0].float().to(device)
        y_src = src_item[1].to(device)

        x_trgt = trgt_item[0].float().to(device)

        p = float(iter_i + start_step) / total_steps
        weights = (2. / (1. + np.exp(-10 * p))) - 1

        x = torch.cat((x_src, x_trgt), dim=0)

        ####################
        #  Supervised part
        ####################
        h = mdl(x)
        y_hat_src = cls(h[:x_src.shape[0], :])
        classification_loss = nn.functional.cross_entropy(y_hat_src, y_src)

        ####################
        #     DA part
        ####################
        if do_rampup:
            y_hat_domain = disc(h, alpha=torch.tensor([weights]).to(device))
        else:
            y_hat_domain = disc(h, alpha=torch.tensor([grl_multiplier]).to(device))

        y_domain = torch.cat((torch.zeros(x_src.shape[0], dtype=torch.long),
                              torch.ones(x_trgt.shape[0], dtype=torch.long)), dim=0).to(device)

        domain_loss = nn.functional.cross_entropy(y_hat_domain, y_domain)

        ####################
        #     ICT part
        ####################
        if current_epoch >= ict_start_epoch:

            with torch.no_grad():
                h_trgt = mdl_ema(x_trgt)
                y_hat_trgt = cls_ema(h_trgt)

                mixed_x_trgt, mixed_y_trgt, _ = mixup_function(x=x_trgt,
                                                               y=y_hat_trgt.detach(),
                                                               beta_param=beta_dist_param)

            if do_conf_thr:
                mask = y_hat_trgt.softmax(dim=1).max(dim=1)[0].ge(conf_thr).float()
                mask_flag = mask.sum()

            h_mixed_trgt = mdl(mixed_x_trgt)
            y_hat_mixed_trgt = cls(h_mixed_trgt)

            if ict_loss.lower() == 'ce':
                if do_conf_thr:
                    if mask_flag > 0.0:
                        consistency_loss = nn.functional.cross_entropy(y_hat_mixed_trgt, mixed_y_trgt.softmax(dim=1),
                                                                       reduction='none')
                        consistency_loss = (mask * consistency_loss).mean()
                    else:
                        consistency_loss = torch.tensor([0.0]).to(device)
                else:
                    consistency_loss = nn.functional.cross_entropy(y_hat_mixed_trgt, mixed_y_trgt.softmax(dim=1))

            elif ict_loss.lower() == 'mse':
                if do_conf_thr:
                    if mask_flag > 0.0:
                        consistency_loss = nn.functional.mse_loss(y_hat_mixed_trgt.softmax(dim=1), mixed_y_trgt.softmax(dim=1),
                                                                  reduction='none').sum(1)
                        consistency_loss = (mask * consistency_loss).mean()
                    else:
                        consistency_loss = torch.tensor([0.0]).to(device)

                else:
                    consistency_loss = nn.functional.mse_loss(y_hat_mixed_trgt.softmax(dim=1), mixed_y_trgt.softmax(dim=1))
            else:
                ValueError()
        else:
            consistency_loss = torch.tensor([0.0]).to(device)

        ####################
        #   Backward pass
        ####################

        if current_epoch < ict_start_epoch:
            loss = classification_loss + domain_loss
        else:
            if do_rampup:
                loss = classification_loss + domain_loss + (consistency_loss * weights)
            else:
                loss = classification_loss + domain_loss + consistency_loss

        opt_.zero_grad()
        loss.backward()
        opt_.step()

        step_ += 1
        mdl_ema, cls_ema = update_ema_model_params(feature_extractor=mdl, ema_feature_extractor=mdl_ema,
                                                   classifier=cls, ema_classifier=cls_ema, ema_rate=ict_alpha,
                                                   global_step=step_)

        ####################
        #   misc
        ####################
        epoch_cls_loss.append(classification_loss.item())
        epoch_da_loss.append(domain_loss.item())
        epoch_cons_loss.append(consistency_loss.item())

        targets.append(y_src.unsqueeze(1).cpu())
        predictions.append(y_hat_src.detach().cpu())

        domain_targets.append(y_domain.unsqueeze(1).cpu())
        domain_predictions.append(y_hat_domain.detach().cpu())

    targets = torch.vstack(targets).squeeze()
    predictions = torch.vstack(predictions)
    predictions = torch.argmax(torch.softmax(predictions, dim=1), dim=1)
    classification_accuracy = (predictions == targets).float().mean()

    domain_targets = torch.vstack(domain_targets).squeeze()
    domain_predictions = torch.vstack(domain_predictions)
    domain_predictions = torch.argmax(torch.softmax(domain_predictions, dim=1), dim=1)
    domain_accuracy = (domain_predictions == domain_targets).float().mean()

    return mdl, cls, disc, np.mean(epoch_cls_loss), np.mean(epoch_da_loss), classification_accuracy, domain_accuracy, \
           step_, np.mean(epoch_cons_loss)


def _validation(data, mdl, cls, device):
    """ validation function

    :param data: validation data
    :param mdl: feature extractor
    :param cls: classifier
    :param device: device GPU or CPU

    :return: feature extractor, classifier, epoch loss, and accuracy
    """

    epoch_loss = []

    targets = []
    predictions = []

    mdl.eval()
    cls.eval()

    with torch.no_grad():
        for item in data:
            x = item[0].float().to(device)
            y = item[1].to(device)

            h = mdl(x)
            y_hat = cls(h)

            loss = nn.functional.cross_entropy(y_hat, y)

            epoch_loss.append(loss.item())
            targets.append(y.unsqueeze(1).cpu())
            predictions.append(y_hat.detach().cpu())

    targets = torch.vstack(targets)
    predictions = torch.vstack(predictions)

    predictions = torch.argmax(torch.softmax(predictions, dim=1), dim=1).unsqueeze(1)

    correct = (predictions == targets).float().sum()

    accuracy = correct / len(targets)

    return mdl, cls, np.mean(epoch_loss), accuracy


def mixup_function(x, y, beta_param):
    """THe function for performing MixUp augmentation

    :param x: the input data
    :param y: the corresponding labels for input data
    :param beta_param: the parameter of Beta distribution

    :return: augmented x, augmented y, and lambda value drawn from Beta distribution
    """

    trgt_batch_size = x.shape[0]

    beta_dist = Beta(torch.tensor([beta_param]), torch.tensor([beta_param]))
    lambda_ = beta_dist.sample().item()

    index = torch.randperm(trgt_batch_size)

    mixed_x = lambda_ * x + (1 - lambda_) * x[index, :]
    mixed_y = lambda_ * y + (1 - lambda_) * y[index, :]

    return mixed_x, mixed_y, lambda_


def update_ema_model_params(feature_extractor, ema_feature_extractor, classifier, ema_classifier, ema_rate, global_step):
    """adapted from ICT original repo: https://github.com/vikasverma1077/ICT
    """

    ema_rate = min(1 - 1 / (global_step + 1), ema_rate)

    for ema_param, param in zip(ema_feature_extractor.parameters(), feature_extractor.parameters()):
        ema_param.data.mul_(ema_rate).add_(param.data, alpha=(1 - ema_rate))

    for ema_param, param in zip(ema_classifier.parameters(), classifier.parameters()):
        ema_param.data.mul_(ema_rate).add_(param.data, alpha=(1 - ema_rate))

    return ema_feature_extractor, ema_classifier

# EOF
