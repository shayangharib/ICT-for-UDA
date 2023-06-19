import numpy as np
from time import time
import os
import torch
from torch import nn

from tools import save_model, print_baseline_results, set_optimizer, get_model, get_data, get_transformation

__author__ = 'Shayan Gharib'
__docformat__ = 'reStructuredText'
__all__ = ['source_only']


def source_only(params, device, index):
    """the baseline system

    :param params: settings
    :param device: device GPU or CPU
    :param index: the index of the corresponding experiment

    :return:
    """

    epochs = params['training']['nb_epochs']
    patience = params['training']['patience']
    early_stopping = params['training']['early_stopping']
    source_type = params['training']['source_dataset_name']

    data_param_dict = {'batch_size': params['data']['batch_size'],
                       'shuffle': params['data']['shuffle'],
                       'drop_last': params['data']['drop_last'],
                       'n_workers': params['data']['nb_workers']
                       }

    source_transform, target_transform = get_transformation(source=params['training']['source_dataset_name'],
                                                            target=params['training']['target_dataset_name'])

    tr_data = get_data(dataset_name=params['training']['source_dataset_name'],
                       other_dataset=params['training']['target_dataset_name'].lower(),
                       transform_function=source_transform, split='train', **data_param_dict)

    val_data = get_data(dataset_name=params['training']['source_dataset_name'],
                        transform_function=source_transform, split='test', **data_param_dict)

    feature_extractor, classifier, _ = get_model(model_type=source_type, device=device,
                                                 target_name=params['training']['target_dataset_name'])

    optimizer = set_optimizer(opt_config=params['training']['optimizer'],
                              model_params=list(feature_extractor.parameters()) + list(classifier.parameters())
                              )

    best_val_loss = 10000
    patience_cnt = 0

    for ep_ in range(epochs):
        start_time = time()

        feature_extractor.train()
        classifier.train()

        feature_extractor, classifier, tr_loss, tr_acc = _training(data=tr_data, mdl=feature_extractor, cls=classifier,
                                                                   opt_=optimizer, device=device,
                                                                   )

        feature_extractor.eval()
        classifier.eval()

        feature_extractor, classifier, val_loss, val_acc = _validation(data=val_data, mdl=feature_extractor,
                                                                       cls=classifier, device=device
                                                                       )

        end_time = time() - start_time

        print_baseline_results(ep=ep_+1, loss_tr=tr_loss, loss_val=val_loss,
                               tr_acc=tr_acc, val_acc=val_acc, time_=end_time)

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt = 0

                best_epoch = ep_ + 1

                save_model(model=feature_extractor, base_dir=os.path.join(
                    params['training']['models']['base_dir_output'], index),
                           model_f_name=params['training']['models']['model_f_name'])
                save_model(model=classifier, base_dir=os.path.join(
                    params['training']['models']['base_dir_output'], index),
                           model_f_name=params['training']['models']['classifier_f_name'])
            else:
                patience_cnt += 1

            if patience_cnt > patience:
                print('\n  -- Epoch: {} -- Best validation loss: {}\n'.format(best_epoch, best_val_loss))
                break

    if not early_stopping:
        save_model(model=feature_extractor, base_dir=os.path.join(
            params['training']['models']['base_dir_output'], index),
                   model_f_name=params['training']['models']['model_f_name'])
        save_model(model=classifier, base_dir=os.path.join(
            params['training']['models']['base_dir_output'], index),
                   model_f_name=params['training']['models']['classifier_f_name'])

        print('\n  -- The training is done. Models (of last epoch) are saved. \n\n')

    return


def _training(data, mdl, cls, device, opt_):
    """training function

    :param data: source training data
    :param mdl: feature extractor
    :param cls: classifier
    :param device: device GPU or CPU
    :param opt_: optimizer

    :return: feature extractor, classifier, epoch loss, and accuracy
    """

    epoch_loss = []

    targets = []
    predictions = []

    for item in data:
        x = item[0].float().to(device)
        y = item[1].to(device)

        h = mdl(x)
        y_hat = cls(h)

        loss = nn.functional.cross_entropy(y_hat, y)

        opt_.zero_grad()
        loss.backward()
        opt_.step()

        epoch_loss.append(loss.item())
        targets.append(y.unsqueeze(1).cpu())
        predictions.append(y_hat.detach().cpu())

    targets = torch.vstack(targets)
    predictions = torch.vstack(predictions)

    predictions = torch.argmax(predictions.softmax(dim=1), dim=1).unsqueeze(1)

    correct = (predictions == targets).sum()
    accuracy = correct / targets.shape[0]

    return mdl, cls, np.mean(epoch_loss), accuracy


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

    predictions = torch.argmax(predictions.softmax(dim=1), dim=1).unsqueeze(1)

    correct = (predictions == targets).sum()

    accuracy = correct / len(targets)

    return mdl, cls, np.mean(epoch_loss), accuracy

# EOF
