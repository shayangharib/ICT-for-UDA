import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .mnistm import MNISTM

__author__ = 'Shayan Gharib'
__docformat__ = 'reStructuredText'
__all__ = ['get_data', 'get_transformation']


def get_data(dataset_name, transform_function, split, batch_size, shuffle, drop_last, n_workers, other_dataset=None):
    """preparing the datasets

    :param dataset_name: The name of the dataset that is going to be loaded [mnist, usps, or mnistm]
    :param transform_function: The transform functions for the data if needed
    :param split: The split set of the dataset to be loaded
    :param batch_size: the size of mini-batch
    :param shuffle: Whether to shuffle the data
    :param drop_last: Whether to drop the last samples of the dataset that do not make a complete mini-batch
    :param n_workers: Number of workers, fix it to a positive value for multiprocess data loading
    :param other_dataset:

    :return: data loader of the dataset
    """

    if dataset_name.lower() == 'mnist':
        if split == 'train':
            dataset = datasets.MNIST(root='./datasets/MNIST/', train=True, download=True, transform=transform_function)

        elif split == 'test':
            dataset = datasets.MNIST(root='./datasets/MNIST/', train=False, download=True,
                                     transform=transform_function)

    elif dataset_name.lower() == 'usps':
        if split == 'train':
            dataset = datasets.USPS(root='./datasets/USPS/', train=True, download=True, transform=transform_function)

        elif split == 'test':
            dataset = datasets.USPS(root='./datasets/USPS/', train=False, download=True,
                                    transform=transform_function)

    elif dataset_name.lower() == 'mnistm':
        if split == 'train':
            dataset = MNISTM(root='./datasets/MNISTM/', processed_root='./datasets/MNISTM/processed/', train=True,
                             transform=transform_function)

        elif split == 'test':
            dataset = MNISTM(root='./datasets/MNISTM/', processed_root='./datasets/MNISTM/processed/', train=False,
                             transform=transform_function)
    else:
        NotImplementedError("The requested dataset has not been implemented yet!!!")

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=shuffle,
        drop_last=drop_last)

    return data_loader


def get_transformation(source, target):
    """The transformations needed for each setup

    :param source: source dataset
    :param target: target dataset

    :return: transformations for both source and target datasets
    """

    if source.lower() == 'mnist' and target.lower() == 'mnistm':
        source_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

        target_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
        ])

    elif source.lower() == 'mnist' and target.lower() == 'usps':
        source_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        target_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
        ])

    elif source.lower() == 'usps' and target.lower() == 'mnist':
        source_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
        ])

        target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    else:
        NotImplementedError('The transformation for the selected setup of source and target datasets does not exist!')

    return source_transform, target_transform
# EOF

