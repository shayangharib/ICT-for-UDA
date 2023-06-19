import numpy as np
import os

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tools import get_model, load_model, get_data, get_transformation


__author__ = 'Shayan Gharib'
__docformat__ = 'reStructuredText'
__all__ = ['evaluation']


def evaluation(params, device, index, load_model_type, confusion_mat_name):
    """ the function to evaluate the model performance on test set of a target domain

    :param params: settings
    :param device: device --> GPU or CPU
    :param index: the experiment index
    :param load_model_type: the method name for loading its corresponding saved models
    :param confusion_mat_name: the name of the file for saving the confusion matrix

    :return:
    """

    data_param_dict = {'batch_size': params['data']['batch_size'],
                       'shuffle': params['data']['shuffle'],
                       'drop_last': False,
                       'n_workers': params['data']['nb_workers']
                       }

    _, target_transform = get_transformation(source=params['training']['source_dataset_name'],
                                             target=params['training']['target_dataset_name'])

    data = get_data(dataset_name=params['training']['target_dataset_name'], transform_function=target_transform,
                    split='test', **data_param_dict)

    plot_dir = os.path.join(params['training']['plot_root_dir'], index)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    feature_extractor, classifier, _ = get_model(model_type=params['training']['source_dataset_name'], device=device,
                                                 target_name=params['training']['target_dataset_name'])

    if load_model_type.lower() == 'dann':
        feature_extractor = load_model(model=feature_extractor,
                                       base_dir=os.path.join(params['training']['models']['base_dir_output'], index),
                                       model_f_name='dann_' + params['training']['models']['model_f_name']
                                       )
        classifier = load_model(model=classifier,
                                base_dir=os.path.join(params['training']['models']['base_dir_output'], index),
                                model_f_name='dann_' + params['training']['models']['classifier_f_name']
                                )

    elif load_model_type.lower() == 'ict':
        feature_extractor = load_model(model=feature_extractor,
                                       base_dir=os.path.join(params['training']['models']['base_dir_output'], index),
                                       model_f_name='ict_' + params['training']['models']['model_f_name']
                                       )
        classifier = load_model(model=classifier,
                                base_dir=os.path.join(params['training']['models']['base_dir_output'], index),
                                model_f_name='ict_' + params['training']['models']['classifier_f_name']
                                )

    elif load_model_type.lower() == 'ict_only':
        feature_extractor = load_model(model=feature_extractor,
                                       base_dir=os.path.join(params['training']['models']['base_dir_output'], index),
                                       model_f_name='ict_only_' + params['training']['models']['model_f_name']
                                       )
        classifier = load_model(model=classifier,
                                base_dir=os.path.join(params['training']['models']['base_dir_output'], index),
                                model_f_name='ict_only_' + params['training']['models']['classifier_f_name']
                                )

    elif load_model_type.lower() == 'source':
        feature_extractor = load_model(model=feature_extractor,
                                       base_dir=os.path.join(params['training']['models']['base_dir_output'], index),
                                       model_f_name=params['training']['models']['model_f_name']
                                       )
        classifier = load_model(model=classifier,
                                base_dir=os.path.join(params['training']['models']['base_dir_output'], index),
                                model_f_name=params['training']['models']['classifier_f_name']
                                )

    else:
        ValueError('The selected model type is unknown !!!')

    y_list = []
    y_hat_list = []

    with torch.no_grad():
        for item in data:
            x = item[0].float().to(device)

            h = feature_extractor(x)
            y_hat = classifier(h)

            y = item[1].unsqueeze(1)

            y_list.extend(y.detach().cpu().numpy())

            y_hat_list.extend(y_hat.softmax(dim=1).argmax(dim=1, keepdim=True).detach().cpu().numpy())

    labels = np.vstack(y_list)
    predictions = np.vstack(y_hat_list)

    nb_correct = (predictions == labels).astype(np.float).sum()
    target_test_accuracy = nb_correct / labels.shape[0]

    print('    -- Accuracy: {:7.4f}'.format(target_test_accuracy))

    cm = confusion_matrix(labels.squeeze(), predictions.squeeze())
    plt.figure()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.savefig(os.path.join(plot_dir, confusion_mat_name), dpi=300, bbox_inches='tight')
    plt.close()

    print('\n    -- Confusion matrix is saved.\n')

# EOF
