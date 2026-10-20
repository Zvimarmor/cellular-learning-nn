"""MNIST / CIFAR-10 dataloaders."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}
SHAPES = {"mnist": (1, 28, 28), "cifar10": (3, 32, 32)}


def get_loaders(name: str, data_dir: str, batch_size: int, num_workers: int, augment: bool,
                pin_memory: bool, download: bool = True):
    name = name.lower()
    mean, std = STATS[name]
    norm = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if name == "mnist":
        train_tf = test_tf = transforms.Compose(norm)
        ds = datasets.MNIST
    else:
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else []
        train_tf, test_tf = transforms.Compose(aug + norm), transforms.Compose(norm)
        ds = datasets.CIFAR10
    train = ds(data_dir, train=True, transform=train_tf, download=download)
    test = ds(data_dir, train=False, transform=test_tf, download=download)
    kw = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    test_loader = DataLoader(test, batch_size=max(batch_size, 256), shuffle=False, **kw)
    return train_loader, test_loader, SHAPES[name], 10
