import os
import shutil
import torch
import torchvision.transforms as T
from torchvision.datasets.vision import VisionDataset
from torchvision.io import read_image
from typing import Any, Callable, Optional, Tuple


__author__ = 'Shayan Gharib'
__docformat__ = 'reStructuredText'
__all__ = ['MNISTM']


class MNISTM(VisionDataset):
    """ MNISTM Dataset.

    Args:
        root (string): Root directory of MNIST-M dataset
        train (bool, optional): If True, creates dataset from training set, otherwise from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    train_folder = 'mnist_m_train/'
    test_folder = 'mnist_m_test/'

    def __init__(
            self,
            root: str,
            processed_root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,

    ) -> None:
        super(MNISTM, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.train = train
        self.processed_root = processed_root

        self.data = []
        self.targets = []

        if not self._check_exists():
            raise RuntimeError('Dataset does not exist.' +
                               ' Please download the MNIST-M dataset manually '
                               '(from: https://drive.google.com/file/d/0B_tExHiYS-0veklUZHFYT19KYjg/view?resourcekey=0-DE7ivVVSn8PwP9fdHvSUfA),'
                               'and place it in the ``MNISTM'' folder.')

        if self.train:
            self.data_folder = self.train_folder
            self.label_file = 'mnist_m_train_labels.txt'
        else:
            self.data_folder = self.test_folder
            self.label_file = 'mnist_m_test_labels.txt'

        if not self._check_processed():
            if not os.path.exists(self.processed_root):
                os.makedirs(self.processed_root)

            shutil.unpack_archive(filename=os.path.join(self.root, 'mnist_m.tar.gz'),
                                  extract_dir=self.processed_root, format='gztar')

        self.processed_root = os.path.join(self.processed_root, 'mnist_m/')

        with open(os.path.join(self.processed_root, self.label_file)) as f:
            for line in f:
                filename, label = line.split()
                image = read_image(path=os.path.join(self.processed_root, self.data_folder, filename))
                assert (image.dtype == torch.uint8)
                assert (image.ndimension() == 3)
                self.data.append(image)
                self.targets.append(int(label))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = T.ToPILImage(mode='RGB')(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.root, 'mnist_m.tar.gz'))

    def _check_processed(self) -> bool:
        return os.path.exists(os.path.join(self.root, 'processed', 'mnist_m'))

# EOF

