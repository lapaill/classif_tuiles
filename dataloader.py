import os

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class HDF5Dataset(Dataset):
    """PyTorch Dataset from HDF5 files"""

    def __init__(self, data_file, labels_file, transform=None):
        """
        Args:
            data_file (string): Path to the HDF5 file with images. The dataset key must be 'x'
            labels_file (string): Path to the HDF5 file with labels. The labels key must be 'y'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = h5py.File(data_file)['x']
        self.labels = h5py.File(labels_file)['y']
        self.transform = transform
        self.classes = [0, 1]  # à modifier

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = Image.fromarray(self.data[idx])
        y = int(self.labels[idx])
        if self.transform:
            x = self.transform(x)
        return (x, y)


class RandomRotate90(object):
    def __init__(self, choice='uniform'):
        self.choice = choice

    def _get_param(self):
        if self.choice == 'uniform':
            k = np.random.choice([0, 90, 180, 270])
        return k

    def __call__(self, img):
        angle = self._get_param()
        return vision_F.rotate(img, angle)


def get_dataloader(datadir, batch_size, pretrained, augmented):
    #    dataset = ImageFolder(
    #        datadir, transform=get_transforms(pretrained, augmented))
    data = os.path.join(datadir, 'data.h5')
    labels = os.path.join(datadir, 'labels.h5')
    dataset = HDF5Dataset(
        data, labels, transform=get_transforms(pretrained, augmented))
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=10)
    return dataloader


def get_transforms(pretrained, augmented):
    if pretrained:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.7364, 0.5600, 0.7052],
                                         std=[0.229, 0.1584, 0.1330])
    if augmented:
        print('Use Augmentation plus')
        trans = transforms.Compose([
            #            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotate90(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    return trans
