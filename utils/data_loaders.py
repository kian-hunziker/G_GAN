import torch
import torchvision
import normflows as nf

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os.path
import utils.rotate_mnist


class RotMnistDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path, transform=transforms.ToTensor(), one_hot_encode=False, no_labels=False, single_class=None):
        self.data_path = data_path
        self.label_path = label_path
        if single_class is None:
            self.data = np.load(self.data_path)
            self.targets = np.load(self.label_path)
        else:
            assert isinstance(single_class, int)
            data = np.load(self.data_path)
            targets = np.load(self.label_path)
            indices = np.where(targets == single_class)
            self.data = data[indices]
            self.targets = targets[indices]

        self.transform = transform
        self.one_hot_encode = one_hot_encode
        self.n_classes = 10
        self.no_labels = no_labels

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        if self.no_labels is True:
            y = torch.zeros(1)
        else:
            if self.one_hot_encode is True:
                y = torch.zeros(self.n_classes)
                y[self.targets[index]] = 1
            else:
                y = self.targets[index]

        return x, y

    def __len__(self):
        return self.data.shape[0]


def get_standard_mnist_dataloader(root,
                                  batch_size=64,
                                  mean=0.5,
                                  std=0.5,
                                  shuffle=True,
                                  img_size=28,
                                  train=True) -> (torch.utils.data.Dataset, torch.utils.data.DataLoader):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(img_size), transforms.Normalize((mean,), (std,))]
    )
    data_path = f'{root}/datasets'
    dataset = datasets.MNIST(root=data_path, train=train, transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, data_loader


def get_rotated_mnist_dataloader(root,
                                 batch_size=64,
                                 mean=0.5,
                                 std=0.5,
                                 num_examples=60000,
                                 num_rotations=8,
                                 max_angle=180,
                                 shuffle=True,
                                 one_hot_encode=False,
                                 no_labels=False,
                                 img_size=28,
                                 train=True,
                                 single_class=None,
                                 glow=False) -> (RotMnistDataset, torch.utils.data.DataLoader):
    # TODO: do we normalize the MNIST dataset?
    if glow is True:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(img_size), nf.utils.Scale(255. / 256.), nf.utils.Jitter(1 / 256.)]
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(img_size), transforms.Normalize((mean,), (std,))]
        )
    # check if dataset already exists
    suffix = f'_{num_examples}ex_{num_rotations}rot_{max_angle}deg'
    if train is True:
        suffix = suffix + '.npy'
    else:
        suffix = suffix + '_test.npy'

    rot_mnist_path = f'{root}/datasets/RotMNIST'
    data_path = f'{rot_mnist_path}/data{suffix}'
    label_path = f'{rot_mnist_path}/labels{suffix}'
    #data_path = root + '/data' + suffix
    #label_path = root + '/labels' + suffix
    if os.path.exists(data_path) and os.path.exists(label_path):
        print(f'Loading data and labels from directories \n      data: {data_path} \n      labels: {label_path}\n')
    else:
        print(f'Generating rotated MNIST training data')
        utils.rotateMNIST.generate_rotated_mnist_dataset(root=root,
                                                         num_examples=num_examples,
                                                         num_rotations=num_rotations,
                                                         max_angle=max_angle,
                                                         train=train)

    dataset = RotMnistDataset(data_path=data_path,
                              label_path=label_path,
                              transform=transform,
                              one_hot_encode=one_hot_encode,
                              no_labels=no_labels,
                              single_class=single_class)
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
