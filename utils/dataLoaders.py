import torch

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os.path
import utils.rotateMNIST


class RotMnistDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path, transform=transforms.ToTensor(), one_hot_encode=False):
        self.data_path = data_path
        self.label_path = label_path
        self.data = np.load(self.data_path)
        self.targets = np.load(self.label_path)
        self.transform = transform
        self.one_hot_encode = one_hot_encode
        self.n_classes = 10

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        if self.one_hot_encode is True:
            y = torch.zeros(self.n_classes)
            y[self.targets[index]] = 1
        else:
            y = self.targets[index]

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
                                 num_examples=60000,
                                 num_rotations=8,
                                 max_angle=180,
                                 shuffle=True,
                                 one_hot_encode=False) -> (RotMnistDataset, torch.utils.data.DataLoader):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    )
    suffix = f'_{num_examples}ex_{num_rotations}rot_{max_angle}deg.npy'
    data_path = root + '/data' + suffix
    label_path = root + '/labels' + suffix
    if os.path.exists(data_path) and os.path.exists(label_path):
        print(f'Loading data and labels from directory: {root}')
    else:
        print(f'Generating rotated MNIST training data')
        utils.rotateMNIST.generate_rotated_mnist_dataset(dir_path=root,
                                                         num_examples=num_examples,
                                                         num_rotations=num_rotations,
                                                         max_angle=max_angle)

    dataset = RotMnistDataset(data_path=data_path,
                              label_path=label_path,
                              transform=transform,
                              one_hot_encode=one_hot_encode)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataset, data_loader


def rot_mnist_test():
    batch_size = 8
    one_hot_encode = True

    dataset, data_loader = get_rotated_mnist_dataloader(root='../datasets/RotMNIST',
                                                        batch_size=batch_size,
                                                        one_hot_encode=one_hot_encode,
                                                        num_examples=60000,
                                                        num_rotations=8,
                                                        max_angle=180)

    #first_batch = next(iter(data_loader))
    #images = first_batch[0]
    #labels = first_batch[1]

    images, labels = next(iter(data_loader))

    print(f'Total number of training examples: {len(dataset)}')
    print(f'data shape: {images.shape}')
    print(f'labels shape: {labels.shape}')
    if one_hot_encode is True:
        print(f'first one hot vec: {labels[0]}')

    fig, axes = plt.subplots(batch_size // 4, 4, figsize=(10, 5))
    for i in range(batch_size):
        ax = axes[i // 4, i % 4]
        ax.imshow(images[i][0], cmap='gray')
        if one_hot_encode is True:
            label = np.where(labels[i] == 1)[0]
        else:
            label = labels[i]
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.show()


#rot_mnist_test()
