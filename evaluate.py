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

from generators import Generator, init_generator_weights_z2
from discriminators import Discriminator, init_discriminator_weights_z2
from utils.dataLoaders import get_rotated_mnist_dataloader
from utils.optimizers import get_optimizers

device = 'cpu'


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen = Generator(gen_arch=checkpoint['gen_arch'])
    disc = Discriminator([1, 28, 28], disc_arch=checkpoint['disc_arch'])
    gen.load_state_dict(checkpoint['generator'])
    disc.load_state_dict(checkpoint['discriminator'])
    gen.to(device)
    disc.to(device)
    gen.eval()
    disc.eval()
    return gen, disc


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


def plot_32_example_digits(digit):
    num_images = 32

    # one hot encoded labels for current class
    label = torch.zeros(num_images, 10)
    label[:, digit] = torch.ones(num_images)
    label = label.to(device)

    # latent noise for image generation
    noise = torch.randn(num_images, 64).to(device)
    # generate images and convert to numpy
    images = gen(noise, label)
    plot_images(images)


def plot_examples_for_all_classes():
    for i in range(10):
        plot_32_example_digits(i)


def plot_slight_variations(digit, inc_size, start_from_zero=True):
    num_images = 32
    label = torch.zeros(num_images, 10)
    label[:, digit] = torch.ones(num_images)
    noise = torch.ones(num_images, 64)
    if start_from_zero is True:
        noise_start = torch.zeros(1, 64)
    else:
        noise_start = torch.randn(1, 64) * 0.5
    noise_inc = F.normalize(torch.randn(1, 64), dim=1) * inc_size
    for i in range(num_images):
        noise[i] = noise_start + i * noise_inc
    print(noise.shape)
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

gen, disc = load_checkpoint('trained_models/p4_rot_mnist/2023-10-17 22:02:15/checkpoint_9000')
#plot_examples_for_all_classes()
for i in range(15):
    inc_size = 0.5 #0.5 * (1 + i)
    plot_slight_variations(7, inc_size=inc_size, start_from_zero=False)
print('loaded')
