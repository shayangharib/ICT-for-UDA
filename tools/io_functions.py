import yaml
import os
import pickle

import torch

__author__ = 'Shayan Gharib'
__docformat__ = 'reStructuredText'
__all__ = ['read_yaml', 'save_model', 'load_model', 'save_pickle_file', 'load_pickle_file', 'save_yaml']


def read_yaml(f_name='settings.yml'):
    """read a yaml settings file.

    :param f_name: the name of the settings yaml file, placed in the same directory as main.py

    :return: parsed yaml file as a python dictionary
    """
    root_dir = './'
    parsed_yaml_file = yaml.load(open(os.path.join(root_dir, f_name)), Loader=yaml.FullLoader)

    return parsed_yaml_file


def save_yaml(param_dict, index):
    """To save the settings yaml file for each experiment if needed

    :param param_dict: the python dictionary containing the settings
    :param index: the index corresponding to the index of an experiment
    :return:
    """
    dir_ = param_dict['training']['yaml_dir']
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    f_name = 'settings_{}.yml'.format(index)

    path = os.path.join(dir_, f_name)

    with open(path, 'w') as yml_f:
        yaml.dump(param_dict, yml_f, default_flow_style=False)

    return


def save_model(model, base_dir, model_f_name):
    """ saving the model architecture of DNNs

    :param model: the DNN models
    :param base_dir: the directory to save the file
    :param model_f_name: the name that will be given to the saved file
    :return:
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    torch.save(model.state_dict(), os.path.join(base_dir, model_f_name))
    return


def load_model(model, base_dir, model_f_name):
    """loading the saved model architecture of DNNs

    :param model: the saved DNN model
    :param base_dir: the directory that the saved file is kept
    :param model_f_name: the file name of the model

    :return: model
    """
    model.load_state_dict(torch.load(os.path.join(base_dir, model_f_name)))

    return model


def load_pickle_file(path):
    """loading a pickle file

    :param path: path to the directory of the pickle file
    :return:
    """

    f = pickle.load(open(path, 'rb'))

    return f


def save_pickle_file(f, path):
    """saving a pickle file

    :param f: the object
    :param path: the desired path for saving the pickle file
    :return:
    """
    pickle.dump(f, open(path, 'wb'))

    return

# EOF

