import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from generators import Generator, init_generator_weights_z2
from discriminators import Discriminator, init_discriminator_weights_z2
from utils.dataLoaders import get_rotated_mnist_dataloader
from utils.optimizers import get_optimizers
import datetime

device = 'cpu'  # 'mps' if torch.backends.mps.is_available() else 'cpu'

BATCH_SIZE = 64
NUM_CLASSES = 10
LATENT_DIM = 64
EPS = 1e-6
GEN_ARCH = 'z2_rot_MNIST'
DISC_ARCH = 'z2_rot_MNIST'
IMG_SHAPE = (1, 28, 28)
beta_1 = 0.0
beta_2 = 0.9
lr_g = 0.0001
lr_d = 0.0004

dataset, data_loader = get_rotated_mnist_dataloader(root='datasets/RotMNIST',
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True)

gen = Generator(n_classes=NUM_CLASSES, gen_arch=GEN_ARCH, latent_dim=LATENT_DIM).to(device)
disc = Discriminator(img_shape=IMG_SHAPE, disc_arch=DISC_ARCH, n_classes=NUM_CLASSES).to(device)

# orthogonal initialization
gen.apply(init_generator_weights_z2)
disc.apply(init_discriminator_weights_z2)

# get optimizers
gen_optim, disc_optim = get_optimizers(lr_g=lr_g,
                                       lr_d=lr_d,
                                       beta1_g=beta_1,
                                       beta2_g=beta_2,
                                       beta1_d=beta_1,
                                       beta2_d=beta_2,
                                       gen=gen,
                                       disc=disc)


def disc_training_step(real_batch, labels, noise_batch, step, eps=EPS):
    fake_batch = gen(noise_batch, labels)

    disc_opinion_real = disc([real_batch, labels])
    disc_opinion_fake = disc([fake_batch, labels])

    real_fake_rel_avg_opinion = (disc_opinion_real - torch.mean(disc_opinion_fake, dim=0))
    fake_real_rel_avg_opinion = (disc_opinion_fake - torch.mean(disc_opinion_real, dim=0))

    # loss, relativistic average loss
    disc_loss = torch.mean(
        - torch.mean(torch.log(torch.sigmoid(real_fake_rel_avg_opinion) + eps), dim=0)
        - torch.mean(torch.log(1 - torch.sigmoid(fake_real_rel_avg_opinion) + eps), dim=0)
    )

    # TODO Gradient Penalty
    gradient_penalty = 0

    total_disc_loss = disc_loss + gradient_penalty

    disc.zero_grad()
    total_disc_loss.backward()
    disc_optim.step()
