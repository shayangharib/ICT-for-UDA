from argparse import ArgumentParser
import torch
import random
import numpy as np

from processes.baseline import source_only
from processes.dann import dann
from processes.ict_only import ict_only
from processes.ict_dann import ict_dann

from processes.evaluation import evaluation

from tools import read_yaml

__author__ = 'Anonymous'
__docformat__ = 'reStructuredText'
__all__ = ['']


def setting_seed(seed):
    """setting the seed of random processes to ensure reproducibility

    :param seed: the seed for generating random numbers
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    """main function

    :return:
    """
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--index', type=str, required=True)
    argument_parser.add_argument('--seed', type=int, required=False, default=8)
    args = argument_parser.parse_args()
    index = args.index
    seed = args.seed

    settings = read_yaml()

    if settings['general_settings']['set_seed']:
        print('The seed for generating random numbers is set to {}'.format(seed), flush=True)
        setting_seed(seed=seed)

    if torch.cuda.is_available():
        device = 'cuda'
        print('The number of GPUs in use is {}'.format(torch.cuda.device_count()), flush=True)
        print('The GPU name is: {}\n\n'.format(torch.cuda.get_device_name(0)), flush=True)
    else:
        device = 'cpu'

    print('Source Dataset: {}'.format(settings['training']['source_dataset_name']), flush=True)
    print('Target Dataset: {}\n\n'.format(settings['training']['target_dataset_name']), flush=True)

    # ******************** #
    # ***** Baseline ***** #
    # ******************** #
    if settings['step']['source_only']:
        print('-- Baseline training:\n', flush=True)
        source_only(params=settings, device=device, index=index)

        print('\n  -- Evaluation of the source model on target test set:')
        evaluation(params=settings, device=device, index=index,
                   load_model_type='source', confusion_mat_name='src_model_confusion_matrix.png'
                   )

    # ************************** #
    # ******* Adaptation ******* #
    # ************************** #
    if settings['step']['dann']:
        print('\n\n-- Domain adaptation - DANN:\n')
        dann(params=settings, device=device, index=index)

        print('\n  -- Evaluation of the adapted model (DANN) on target test set:')
        evaluation(params=settings, device=device, index=index,
                   load_model_type='dann', confusion_mat_name='trgt_dann_model_confusion_matrix.png')

    if settings['step']['ict_dann']:
        print('\n\n-- Domain adaptation - ICT-DANN:\n')
        ict_dann(params=settings, device=device, index=index)

        print('\n  -- Evaluation of the adapted model (ICT-DANN) on target test set:')
        evaluation(params=settings, device=device, index=index,
                   load_model_type='ict', confusion_mat_name='trgt_ict_model_confusion_matrix.png')

    if settings['step']['ict_only']:
        print('\n\n-- Domain adaptation - ICT-only:\n')
        ict_only(params=settings, device=device, index=index)

        print('\n  -- Evaluation of the adapted model (ICT-only) on target test set:')
        evaluation(params=settings, device=device, index=index,
                   load_model_type='ict_only', confusion_mat_name='trgt_ict_only_model_confusion_matrix.png')


if __name__ == "__main__":
    main()

# EOF
