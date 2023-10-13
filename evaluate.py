import os

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import random
import datetime
from time import time
from tqdm import tqdm

from generators import Generator, init_generator_weights_z2
from discriminators import Discriminator, init_discriminator_weights_z2
from utils.dataLoaders import get_rotated_mnist_dataloader
from utils.optimizers import get_optimizers


gen = Generator()
gen.load_state_dict(torch.load('trained_models/z2_rot_mnist/2023-10-13 16:05:35/generator_test'))
print(gen)