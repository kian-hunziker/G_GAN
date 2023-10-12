import torch

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os.path
import utils.rotateMNIST


class RotMnistDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=transforms.ToTensor()):
        self.data_path = root
        self.data = np.load(self.data_path + '/images.npy')
        self.targets = np.load(self.data_path + '/labels.npy')
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.data.shape[0]


def get_standard_mnist_dataloader(root='datasets/',
                                  batch_size=64,
                                  mean=0.5,
                                  std=0.5,
                                  shuffle=True) -> (torch.utils.data.Dataset, torch.utils.data.DataLoader):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    )

    dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, data_loader


def get_rotated_mnist_dataloader(root='datasets/RotMNIST',
                                 batch_size=64,
                                 mean=0.5,
                                 std=0.5,
                                 shuffle=True) -> (torch.utils.data.Dataset, torch.utils.data.DataLoader):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    )

    if os.path.exists(root + '/images.npy') and os.path.exists(root + '/labels.npy'):
        print(f'loading data and labels from directory {root}')
    else:
        print(f'generating rotated MNIST training data')
        utils.rotateMNIST.generate_rotated_mnist_dataset(dir_path=root, num_examples=60000, num_rotations=8, max_angle=180)

    dataset = RotMnistDataset(root=root, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataset, data_loader


def rot_mnist_test():
    batch_size = 8

    dataset, data_loader = get_rotated_mnist_dataloader(root='../datasets/RotMNIST')

    first_batch = next(iter(data_loader))
    images = first_batch[0]
    labels = first_batch[1]

    fig, axes = plt.subplots(batch_size // 4, 4, figsize=(10, 5))
    for i in range(batch_size):
        ax = axes[i // 4, i % 4]
        ax.imshow(images[i][0], cmap='gray')
        ax.set_title(f"Label: {labels[i]}")
        ax.axis('off')
    plt.show()


