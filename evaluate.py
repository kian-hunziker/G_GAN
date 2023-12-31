import os

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter

import random
import datetime
from time import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from generators import Generator
from discriminators import Discriminator
from utils.checkpoints import load_gen_disc_from_checkpoint, load_checkpoint, print_checkpoint
from utils.data_loaders import get_rotated_mnist_dataloader
from utils.optimizers import get_optimizers

device = 'cpu'


def plot_images(images: torch.Tensor):
    num_images = images.shape[0]
    images = images.cpu().detach().numpy()

    num_cols = 8
    fig, axes = plt.subplots(num_images // num_cols, num_cols, figsize=(9, 4))
    for i in range(num_images):
        ax = axes[i // num_cols, i % num_cols]
        ax.imshow(images[i][0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_32_example_digits(digit, l_dim=64):
    num_images = 32

    # one hot encoded labels for current class
    label = torch.zeros(num_images, 10)
    label[:, digit] = torch.ones(num_images)
    label = label.to(device)

    # latent noise for image generation
    noise = torch.randn(num_images, l_dim).to(device)
    # generate images and convert to numpy
    images = gen(noise, label)
    plot_images(images)


def plot_32_random_examples(l_dim=64):
    assert 'no_label' in gen.gen_arch
    # latent noise for image generation
    noise = torch.randn(32, l_dim).to(device)
    # generate images and convert to numpy
    images = gen(noise, None)
    plot_images(images)


def plot_examples_for_all_classes(l_dim=64):
    for i in range(10):
        plot_32_example_digits(i, l_dim=l_dim)


def plot_slight_variations(digit, inc_size, l_dim=64, start_from_zero=True):
    num_images = 32
    label = torch.zeros(num_images, 10)
    label[:, digit] = torch.ones(num_images)
    noise = torch.ones(num_images, l_dim)
    if start_from_zero is True:
        noise_start = torch.zeros(1, l_dim)
    else:
        noise_start = torch.randn(1, l_dim) * 0.5
    noise_inc = F.normalize(torch.randn(1, l_dim), dim=1) * inc_size
    for i in range(num_images):
        noise[i] = noise_start + i * noise_inc
    noise = noise.to(device)
    label = label.to(device)
    images = gen(noise, label)
    plot_images(images)


'''
gen = Generator()
gen.load_state_dict(
    torch.load('trained_models/z2_rot_mnist/2023-10-13 18:23:50/generator_test',
               map_location=torch.device('cpu'))
)
gen.eval()

for _ in range(10):
    plot_slight_variations(digit=3, inc_size=0.6, start_from_zero=True)
'''
# plot_examples_for_all_classes()
p4_path = 'trained_models/p4_rot_mnist/2023-11-09_18:14:22/checkpoint_4000'
z2_path = 'trained_models/z2_rot_mnist/2023-10-31 12:34:55/checkpoint_20000'
gen, disc = load_gen_disc_from_checkpoint(checkpoint_path=p4_path, device=device)
checkpoint = load_checkpoint(p4_path, device='cpu')
print_checkpoint(checkpoint)
latent_dim = checkpoint['latent_dim']

plot_examples_for_all_classes(l_dim=latent_dim)
#plot_32_random_examples()
for i in range(4):
    inc_size = 0.5 #0.5 * (1 + i)
    plot_slight_variations(8, l_dim=latent_dim, inc_size=inc_size, start_from_zero=False)
