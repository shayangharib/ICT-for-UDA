import numpy as np
from time import time
import os
import torch
from torch import nn
from torch.distributions import Beta
from itertools import cycle

from tools import save_model, print_ict_only_results, set_optimizer, get_model, get_data, get_transformation

__author__ = 'Shayan Gharib'
__docformat__ = 'reStructuredText'
__all__ = ['ict_only']


def ict_only(params, device, index):
    """the implementation of interpolation consistency training (ICT)

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

    src_val_data = get_data(dataset_name=params['training']['source_dataset_name'], transform_function=source_transform,
                            split='test', **data_param_dict)

    trgt_tr_data = get_data(dataset_name=params['training']['target_dataset_name'],
                            other_dataset=params['training']['source_dataset_name'].lower(),
                            transform_function=target_transform, split='train', **data_param_dict)

    epochs = params['training']['nb_epochs']
    patience = params['training']['patience']
    early_stopping = params['training']['early_stopping']

    source_type = params['training']['source_dataset_name']

    mixup_beta_dist_param = params['training']['mixup']['beta_parameter']

    ict_alpha = params['training']['ict']['alpha']
    rampup_start = params['training']['ict']['ramp_up_start']
    rampup_end = params['training']['ict']['ramp_up_end']

    feature_extractor, classifier, _ = get_model(model_type=source_type, device=device,
                                                 target_name=params['training']['target_dataset_name'])
    feature_extractor_ema, classifier_ema, _ = get_model(model_type=source_type, device=device,
                                                         target_name=params['training']['target_dataset_name'],
                                                         ema_model=True)

    optimizer = set_optimizer(opt_config=params['training']['optimizer'],
                              model_params=list(feature_extractor.parameters()) + list(classifier.parameters())
                              )

    best_val_loss = 10000
    patience_cnt = 0
    best_epoch = 0

    current_step = 0

    for ep_ in range(epochs):
        start_time = time()

        feature_extractor, classifier, tr_cls_loss, consistency_loss, tr_cls_acc, current_step, cons_weight = _train(
            src_data=src_tr_data, trgt_data=trgt_tr_data, mdl=feature_extractor, cls=classifier,
            mdl_ema=feature_extractor_ema, cls_ema=classifier_ema, opt_=optimizer, device=device,
            beta_dist_param=mixup_beta_dist_param, ema_step=current_step, current_epoch=ep_,
            ict_alpha=ict_alpha, rampup_st=rampup_start, rampup_end=rampup_end
        )

        val_loss, val_acc = _validate(data=src_val_data, mdl=feature_extractor, cls=classifier, device=device)

        end_time = time() - start_time

        print_ict_only_results(ep=ep_+1, loss_tr=tr_cls_loss, loss_val=val_loss, loss_consistency=consistency_loss,
                               tr_acc=tr_cls_acc, val_acc=val_acc, cons_weight=cons_weight, time_=end_time)

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt = 0

                best_epoch = ep_ + 1

                save_model(model=feature_extractor, base_dir=os.path.join(
                    params['training']['models']['base_dir_output'], index),
                           model_f_name='ict_only_' + params['training']['models']['model_f_name'])
                save_model(model=classifier, base_dir=os.path.join(
                    params['training']['models']['base_dir_output'], index),
                           model_f_name='ict_only_' + params['training']['models']['classifier_f_name'])
            else:
                patience_cnt += 1

            if patience_cnt > patience:
                print('\n  -- Epoch: {} -- Best validation loss: {}\n\n'.format(best_epoch, best_val_loss))
                break

    if not early_stopping:
        save_model(model=feature_extractor, base_dir=os.path.join(
            params['training']['models']['base_dir_output'], index),
                   model_f_name='ict_only_' + params['training']['models']['model_f_name'])
        save_model(model=classifier, base_dir=os.path.join(
            params['training']['models']['base_dir_output'], index),
                   model_f_name='ict_only_' + params['training']['models']['classifier_f_name'])

        print('\n  -- The training is done. Models (of the last epoch) are saved. \n\n')
    return


def _train(src_data, trgt_data, mdl, cls, mdl_ema, cls_ema, device, opt_, beta_dist_param, ema_step, current_epoch,
           ict_alpha, rampup_st, rampup_end):
    """ICT training function

    :param src_data: source training data
    :param trgt_data: target training data
    :param mdl: feature extractor student
    :param cls: classifier student
    :param mdl_ema: feature extractor teacher
    :param cls_ema: classifier teacher
    :param device: device GPU or CPU
    :param opt_: optimizer
    :param beta_dist_param: the parameter for Beta distribution
    :param ema_step: the current step for ema optimization
    :param current_epoch: current epoch
    :param ict_alpha: ICT EMA rate
    :param rampup_st: starting point for ramping up the weight for ICT
    :param rampup_end: the end point where ramping up the ICT reaches its maximum

    :return:feature extractor, classifier, classification loss, consistency loss, classification accuracy,
    current step for ema optimization, and consistency loss weight
    """

    epoch_cls_loss = []
    epoch_cons_loss = []

    targets = []
    predictions = []

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

        ####################
        #  Supervised part
        ####################
        h_src = mdl(x_src)
        y_hat_src = cls(h_src)

        classification_loss = nn.functional.cross_entropy(y_hat_src, y_src)

        ####################
        #     ICT part
        ####################
        with torch.no_grad():
            h_trgt = mdl_ema(x_trgt)
            y_hat_trgt = cls_ema(h_trgt)

        mix_x_trgt, mix_y_trgt, _ = mixup_function(x=x_trgt, y=y_hat_trgt.detach(), beta_param=beta_dist_param)

        h_mix_trgt = mdl(mix_x_trgt)
        y_hat_mix_trgt = cls(h_mix_trgt)

        consistency_loss = nn.functional.mse_loss(y_hat_mix_trgt.softmax(dim=1), mix_y_trgt.softmax(dim=1))

        ####################
        #   Backward pass
        ####################
        consistency_weight = get_current_consistency_weight(100, current_epoch, iter_i,
                                                            max(len(src_data), len(trgt_data)),
                                                            rampup_st, rampup_end)

        loss = classification_loss + (consistency_loss * consistency_weight)

        opt_.zero_grad()
        loss.backward()
        opt_.step()

        mdl_ema.zero_grad()
        cls_ema.zero_grad()
        ema_step += 1
        mdl_ema, cls_ema = update_ema_model_params(feature_extractor=mdl, ema_feature_extractor=mdl_ema,
                                                   classifier=cls, ema_classifier=cls_ema, ema_rate=ict_alpha,
                                                   global_step=ema_step)
        ####################
        ####################

        epoch_cls_loss.append(classification_loss.item())
        epoch_cons_loss.append(consistency_loss.item())

        targets.append(y_src.unsqueeze(1).cpu())
        predictions.append(y_hat_src.detach().cpu())

    targets = torch.vstack(targets).squeeze()

    predictions = torch.vstack(predictions)
    predictions = torch.argmax(torch.softmax(predictions, dim=1), dim=1)

    classification_accuracy = (predictions == targets).float().mean()

    return mdl, cls, np.mean(epoch_cls_loss), np.mean(epoch_cons_loss), classification_accuracy, ema_step, \
           consistency_weight


def _validate(data, mdl, cls, device):
    """ validation function

    :param data: validation data
    :param mdl: feature extractor
    :param cls: classifier
    :param device: device GPU or CPU

    :return: epoch loss and accuracy
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

    return np.mean(epoch_loss), accuracy


def sigmoid_rampup(current, rampup_length):
    """ adopted from ICT original repo: https://github.com/vikasverma1077/ICT
    """

    # Exponential rampup from https://arxiv.org/abs/1610.02242
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(final_consistency_weight, epoch, step_in_epoch, total_steps_in_epoch,
                                   rampup_st, rampup_end):
    """ adopted from ICT original repo: https://github.com/vikasverma1077/ICT
    """

    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    epoch = epoch - rampup_st
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    return final_consistency_weight * sigmoid_rampup(epoch, rampup_end - rampup_st)


def mixup_function(x, y, beta_param):
    """THe function for performing MixUp augmentation

    :param x: the input data
    :param y: the corresponding labels for input data
    :param beta_param: the parameter of Beta distribution

    :return: augmented x, augmented y, and lambda value drawn from Beta distribution
    """

    batch_size = x.shape[0]

    beta_dist = Beta(torch.tensor([beta_param]), torch.tensor([beta_param]))
    lambda_ = beta_dist.sample().item()

    index = torch.randperm(batch_size)

    mixed_x = lambda_ * x + (1 - lambda_) * x[index, :]
    mixed_y = lambda_ * y + (1 - lambda_) * y[index, :]

    return mixed_x, mixed_y, lambda_


def update_ema_model_params(feature_extractor, ema_feature_extractor, classifier, ema_classifier, ema_rate, global_step):
    """ adapted from ICT original repo: https://github.com/vikasverma1077/ICT
    """

    ema_rate = min(1 - 1 / (global_step + 1), ema_rate)

    for ema_param, param in zip(ema_feature_extractor.parameters(), feature_extractor.parameters()):
        ema_param.data.mul_(ema_rate).add_(param.data, alpha=(1 - ema_rate))

    for ema_param, param in zip(ema_classifier.parameters(), classifier.parameters()):
        ema_param.data.mul_(ema_rate).add_(param.data, alpha=(1 - ema_rate))

    return ema_feature_extractor, ema_classifier

# EOF
