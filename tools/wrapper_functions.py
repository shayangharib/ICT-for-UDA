from torch import optim

from modules.mnist import *

__author__ = 'Shayan Gharib'
__docformat__ = 'reStructuredText'
__all__ = ['set_optimizer', 'get_model']


def set_optimizer(opt_config, model_params):
    """ setting the optimizer

    :param opt_config: the optimizer settings as python dictionary
    :param model_params: the parameters of DNN model

    :return: optimizer
    """

    opt_type = opt_config['type'].lower()

    if opt_type == 'sgd':
        optimizer = optim.SGD(params=model_params, lr=opt_config['lr'], momentum=opt_config['momentum'])

    elif opt_type == 'adam':
        optimizer = optim.Adam(params=model_params, lr=opt_config['lr'], betas=(opt_config['beta_1'],
                                                                                opt_config['beta_2']),
                               )

    elif opt_type == 'rmsprop':
        optimizer = optim.RMSprop(params=model_params, lr=opt_config['lr'])

    else:
        raise NotImplementedError('The assigned type of optimizer is not implemented!')

    return optimizer


def get_model(model_type, device, target_name, nb_disc_output=2, ema_model=False):
    """

    :param model_type: type of the model which will be assigned based on the source dataset
    :param device: device that the experiment is carried on
    :param target_name: the name of the target dataset
    :param nb_disc_output: the number of outputs for discriminator, must be fixed to 2 unless the
    architecture of the model is modified accordingly
    :param ema_model: whether the model is going to be optimized through EMA

    :return: the DNN models as feature extractor, classifier and domain discriminator
    """

    if model_type.lower() == 'mnist':
        if target_name.lower() == 'usps':
            input_channels = 1
        else:
            input_channels = 3

        feature_extractor = MNISTFeatureExtractor(input_channels=input_channels).to(device)
        classifier = MNISTClassifier().to(device)
        discriminator = MNISTDiscriminator(nb_output=nb_disc_output).to(device)

    elif model_type.lower() == 'usps':
        feature_extractor = MNISTFeatureExtractor(input_channels=1).to(device)
        classifier = MNISTClassifier().to(device)
        discriminator = MNISTDiscriminator(nb_output=nb_disc_output).to(device)

    else:
        NotImplementedError('The requested model type does not exist!')

    # if ema_model:
    #     for fe_param in feature_extractor.parameters():
    #         fe_param.requires_grad = False
    #     for c_param in classifier.parameters():
    #         c_param.requires_grad = False

    return feature_extractor, classifier, discriminator
# EOF
