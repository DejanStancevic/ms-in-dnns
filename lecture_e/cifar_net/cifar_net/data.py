import os
import urllib
import zipfile
import pathlib as pl

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import lightning as L


class CifarDataModule(L.LightningDataModule):
    def __init__(self, data_root, batch_size, augment):
        super().__init__()

        self.path = data_root

        if augment:
            self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.2),
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.val_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )
        else:
            self.train_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )
            self.val_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )

        self.batch_size = batch_size

    def prepare_data(self):
        torchvision.datasets.CIFAR10(self.path, train = True, download = True)
        torchvision.datasets.CIFAR10(self.path, train = False, download = True)

    def setup(self, stage):
        self.train_dataset = torchvision.datasets.CIFAR10(self.path, train = True, transform = self.train_transform)
        self.val_dataset = torchvision.datasets.CIFAR10(self.path, train = False, transform = self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return self.val_dataloader()