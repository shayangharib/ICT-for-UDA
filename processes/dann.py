import numpy as np
from time import time
import os
import torch
from torch import nn
from itertools import cycle

from tools import save_model, print_mixup_ict_results, set_optimizer, get_model, get_data, get_transformation

__author__ = 'Shayan Gharib'
__docformat__ = 'reStructuredText'
__all__ = ['dann']


def dann(params, device, index):
    """the gradient reversal method (i.e. DANN)

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

    feature_extractor, classifier, discriminator = get_model(model_type=source_type, device=device,
                                                             target_name=params['training']['target_dataset_name'])

    optimizer = set_optimizer(opt_config=params['training']['optimizer'],
                              model_params=list(feature_extractor.parameters()) + list(classifier.parameters()) + list(
                                  discriminator.parameters())
                              )

    best_val_loss = 10000
    patience_cnt = 0

    nb_iter = min(len(src_tr_data), len(trgt_tr_data))
    total_iterations = epochs * nb_iter

    for ep_ in range(epochs):
        start_time = time()

        start_step = ep_ * nb_iter

        feature_extractor.train()
        classifier.train()
        discriminator.train()

        feature_extractor, classifier, discriminator, tr_cls_loss, tr_da_loss, tr_cls_acc, tr_da_acc = _training(
            src_data=src_tr_data, trgt_data=trgt_tr_data, mdl=feature_extractor, cls=classifier,
            disc=discriminator, opt_=optimizer, device=device, start_step=start_step, total_steps=total_iterations
        )

        feature_extractor.eval()
        classifier.eval()

        feature_extractor, classifier, val_loss, val_acc = _validation(data=src_val_data,
                                                                       mdl=feature_extractor, cls=classifier,
                                                                       device=device
                                                                       )

        end_time = time() - start_time

        print_mixup_ict_results(ep=ep_+1, loss_tr=tr_cls_loss, loss_val=val_loss, loss_domain=tr_da_loss,
                                tr_acc=tr_cls_acc, val_acc=val_acc, acc_domain=tr_da_acc, time_=end_time
                                )

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt = 0

                best_epoch = ep_ + 1

                save_model(model=feature_extractor, base_dir=os.path.join(
                    params['training']['models']['base_dir_output'], index),
                           model_f_name='dann_' + params['training']['models']['model_f_name'])
                save_model(model=classifier, base_dir=os.path.join(
                    params['training']['models']['base_dir_output'], index),
                           model_f_name='dann_' + params['training']['models']['classifier_f_name'])
            else:
                patience_cnt += 1

            if patience_cnt > patience:
                print('\n  -- Epoch: {} -- Best validation loss: {}\n\n'.format(best_epoch, best_val_loss))
                break

    if not early_stopping:
        save_model(model=feature_extractor, base_dir=os.path.join(
            params['training']['models']['base_dir_output'], index),
                   model_f_name='dann_' + params['training']['models']['model_f_name'])
        save_model(model=classifier, base_dir=os.path.join(
            params['training']['models']['base_dir_output'], index),
                   model_f_name='dann_' + params['training']['models']['classifier_f_name'])

        print('\n  -- The training is done. Models (of last epoch) are saved. \n\n')

    return


def _training(src_data, trgt_data, mdl, cls, disc, device, opt_, start_step, total_steps):
    """

    :param src_data: source training data
    :param trgt_data: target training data
    :param mdl: feature extractor
    :param cls: classifier
    :param disc: domain discriminator (domain classifier)
    :param device: device GPU or CPU
    :param opt_: optimizer
    :param start_step: current iteration
    :param total_steps: total number of iterations

    :return: feature extractor, classifier discriminator, classification loss, domain loss, classification accuracy,
    domain accuracy
    """

    epoch_cls_loss = []
    epoch_da_loss = []

    targets = []
    predictions = []

    domain_targets = []
    domain_predictions = []

    src_len = len(src_data)
    trgt_len = len(trgt_data)

    for_loop_zip = zip(src_data, cycle(trgt_data)) if src_len > trgt_len else zip(cycle(src_data), trgt_data)

    for iter_idx, (src_item, trgt_item) in enumerate(for_loop_zip):
        p = float(iter_idx + start_step) / total_steps
        lamb = (2. / (1. + np.exp(-10 * p))) - 1

        x_src = src_item[0].float().to(device)
        y_src = src_item[1].to(device)

        x_trgt = trgt_item[0].float().to(device)

        x = torch.cat((x_src, x_trgt), dim=0)
        y_domain = torch.cat((torch.zeros(x_src.shape[0], dtype=torch.int64),
                              torch.ones(x_trgt.shape[0], dtype=torch.int64)), dim=0).to(device)

        h = mdl(x)
        y_hat_src = cls(h[:x_src.shape[0], :])
        y_hat_domain = disc(h, alpha=torch.tensor([lamb]).to(device))

        classification_loss = nn.functional.cross_entropy(y_hat_src, y_src)
        domain_loss = nn.functional.cross_entropy(y_hat_domain, y_domain)

        loss = classification_loss + domain_loss

        opt_.zero_grad()
        loss.backward()
        opt_.step()

        epoch_cls_loss.append(classification_loss.item())
        epoch_da_loss.append(domain_loss.item())

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

    return mdl, cls, disc, np.mean(epoch_cls_loss), np.mean(epoch_da_loss), classification_accuracy, domain_accuracy


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

# EOF
