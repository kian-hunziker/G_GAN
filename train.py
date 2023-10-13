from time import time

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
from tqdm import tqdm

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

EPOCHS = 5
BATCH_SIZE = 64
NUM_CLASSES = 10
LATENT_DIM = 64
EPS = 1e-6
DISC_UPDATE_STEPS = 2
GEN_ARCH = 'z2_rot_mnist'
DISC_ARCH = 'z2_rot_mnist'
IMG_SHAPE = (1, 28, 28)
beta_1 = 0.0
beta_2 = 0.9
lr_g = 0.0001
lr_d = 0.0004

dataset, data_loader = get_rotated_mnist_dataloader(root='datasets/RotMNIST',
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    one_hot_encode=True)

gen = Generator(n_classes=NUM_CLASSES, gen_arch=GEN_ARCH, latent_dim=LATENT_DIM)
disc = Discriminator(img_shape=IMG_SHAPE, disc_arch=DISC_ARCH, n_classes=NUM_CLASSES)

# orthogonal initialization
gen.apply(init_generator_weights_z2)
disc.apply(init_discriminator_weights_z2)
gen = gen.to(device)
disc = disc.to(device)

# get optimizers
gen_optim, disc_optim = get_optimizers(lr_g=lr_g,
                                       lr_d=lr_d,
                                       beta1_g=beta_1,
                                       beta2_g=beta_2,
                                       beta1_d=beta_1,
                                       beta2_d=beta_2,
                                       gen=gen,
                                       disc=disc)


def get_opinions_for_rel_avg_loss(real_batch, labels, noise_batch):
    # TODO: is it inefficient to move this function out here?
    fake_batch = gen(noise_batch, labels)

    disc_opinion_real = disc([real_batch, labels])
    disc_opinion_fake = disc([fake_batch, labels])

    real_fake_rel_avg_opinion = (disc_opinion_real - torch.mean(disc_opinion_fake, dim=0))
    fake_real_rel_avg_opinion = (disc_opinion_fake - torch.mean(disc_opinion_real, dim=0))

    return real_fake_rel_avg_opinion, fake_real_rel_avg_opinion


def gen_training_step(real_batch, labels, noise_batch, step, eps=EPS):
    real_fake_rel_avg_opinion, fake_real_rel_avg_opinion = get_opinions_for_rel_avg_loss(real_batch,
                                                                                         labels,
                                                                                         noise_batch)

    # loss, relativistic average loss
    gen_loss = torch.mean(
        - torch.mean(torch.log(torch.sigmoid(fake_real_rel_avg_opinion) + eps), dim=0)
        - torch.mean(torch.log(1 - torch.sigmoid(real_fake_rel_avg_opinion) + eps), dim=0)
    )

    gen.zero_grad()
    gen_loss.backward()
    gen_optim.step()


def disc_training_step(real_batch, labels, noise_batch, step, eps=EPS):
    real_fake_rel_avg_opinion, fake_real_rel_avg_opinion = get_opinions_for_rel_avg_loss(real_batch,
                                                                                         labels,
                                                                                         noise_batch)

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


# ---------------------------------------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------------------------------------
for epoch in range(EPOCHS):
    #print(f'EPOCH [{epoch} / {EPOCHS}]')
    start_time = time()
    progbar = tqdm(data_loader)

    examples_per_iteration = BATCH_SIZE * (DISC_UPDATE_STEPS + 1)
    steps_per_epoch = int(len(dataset) // examples_per_iteration)

    # loop through dataset
    for i in range(steps_per_epoch):

        # update discriminator
        for j in range(DISC_UPDATE_STEPS):
            real, labels = next(iter(data_loader))
            noise = torch.randn((real.shape[0], LATENT_DIM))
            real = real.to(device)
            labels = labels.to(device)
            noise = noise.to(device)


            disc_training_step(real_batch=real,
                               labels=labels,
                               noise_batch=noise,
                               step=(i + epoch * steps_per_epoch)
                               )

        # update generator
        real, labels = next(iter(data_loader))
        noise = torch.randn((real.shape[0], LATENT_DIM))
        real = real.to(device)
        labels = labels.to(device)
        noise = noise.to(device)

        gen_training_step(real_batch=real,
                          labels=labels,
                          noise_batch=noise,
                          step=(i + epoch * steps_per_epoch)
                          )

        progbar.update(DISC_UPDATE_STEPS + 1)
        progbar.set_description(f'EPOCH [{epoch} / {EPOCHS}]')

    print(f'Time required for epoch: {((time() - start_time) / 60):.2f}')
