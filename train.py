import os

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import random
import datetime
from time import time
from tqdm import tqdm

from generators import Generator, init_generator_weights_z2
from discriminators import Discriminator, init_discriminator_weights_z2
from utils.dataLoaders import get_rotated_mnist_dataloader
from utils.optimizers import get_optimizers


# TODO name networks properly
IDENTIFIER_FOR_SAVING = 'test'

# setup device
device = 'cpu'
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')

print('-' * 32 + '\n')
print(f'Using device: {device}')

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 64
NUM_CLASSES = 10
LATENT_DIM = 64
EPS = 1e-6
DISC_UPDATE_STEPS = 2
GP_STRENGTH = 0.1
GEN_ARCH = 'p4_rot_mnist'#'z2_rot_mnist'
DISC_ARCH = 'p4_rot_mnist'#'z2_rot_mnist'
IMG_SHAPE = (1, 28, 28)
beta_1 = 0.0
beta_2 = 0.9
lr_g = 0.0001
lr_d = 0.0004

# fix random seeds
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# setup data loader
dataset, data_loader = get_rotated_mnist_dataloader(root='datasets/RotMNIST',
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    one_hot_encode=True,
                                                    num_examples=60000,
                                                    num_rotations=8)
print(f'Total number of training examples: {len(dataset)}')
print(f'Training data path: {dataset.data_path}')

# setup generator and discriminator
gen = Generator(n_classes=NUM_CLASSES, gen_arch=GEN_ARCH, latent_dim=LATENT_DIM)
disc = Discriminator(img_shape=IMG_SHAPE, disc_arch=DISC_ARCH, n_classes=NUM_CLASSES)

n_trainable_params_gen = sum(p.numel() for p in gen.parameters() if p.requires_grad)
n_trainable_params_disc = sum(p.numel() for p in disc.parameters() if p.requires_grad)
print(f'Trainable parameters in Generator: {n_trainable_params_gen}')
print(f'Trainable parameters in Discriminator: {n_trainable_params_disc}\n')

# orthogonal initialization
print('Initializing Generator Weights')
gen.apply(init_generator_weights_z2)
print('Initializing Discriminator Weights')
disc.apply(init_discriminator_weights_z2)

gen = gen.to(device)
disc = disc.to(device)
gen.train()
disc.train()

# get optimizers
gen_optim, disc_optim = get_optimizers(lr_g=lr_g,
                                       lr_d=lr_d,
                                       beta1_g=beta_1,
                                       beta2_g=beta_2,
                                       beta1_d=beta_1,
                                       beta2_d=beta_2,
                                       gen=gen,
                                       disc=disc)

# setup summary writer
current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_dir = f'runs/{GEN_ARCH}/{current_date}'
summ_writer = SummaryWriter(log_dir)
# create fixed latent noise
fixed_noise = torch.randn(32, LATENT_DIM).to(device)
# create fixed random labels
rand_labels = torch.randint(0, 10, (32,))
fixed_labels = torch.zeros(32, 10)
fixed_labels.scatter_(1, rand_labels.unsqueeze(1), 1)

print(f'The fixed labels are: \n {fixed_labels.view(4, 8).numpy()}')

n_steps_for_summary = 10


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

    if step % n_steps_for_summary == 0:
        summ_writer.add_scalar(
            'Generator Loss', gen_loss, global_step=step
        )


def disc_training_step(real_batch, labels, noise_batch, step, disc_step, eps=EPS):
    real_fake_rel_avg_opinion, fake_real_rel_avg_opinion = get_opinions_for_rel_avg_loss(real_batch,
                                                                                         labels,
                                                                                         noise_batch)

    disc_loss = torch.mean(
        - torch.mean(torch.log(torch.sigmoid(real_fake_rel_avg_opinion) + eps), dim=0)
        - torch.mean(torch.log(1 - torch.sigmoid(fake_real_rel_avg_opinion) + eps), dim=0)
    )

    # TODO check Gradient Penalty
    new_real_batch = 1.0 * real_batch
    new_real_batch.requires_grad = True
    new_labels = 1.0 * labels
    new_real_batch = new_real_batch.to(device)
    new_labels = new_labels.to(device)
    disc_opinion_real_new = disc([new_real_batch, new_labels])

    gradient = torch.autograd.grad(
        inputs=new_real_batch,
        outputs=disc_opinion_real_new,
        grad_outputs=torch.ones_like(disc_opinion_real_new),
        create_graph=True,
        retain_graph=True
    )[0]
    # gradient.shape: [batch_size, channels, height, width]
    gradient = gradient.view(gradient.shape[0], -1)
    # gradient.shape: [batch_size, channels * height * width]
    gradient_squared = torch.square(gradient)
    #gradient_norm = gradient.norm(2, dim=1)
    grad_square_sum = torch.sum(gradient_squared, dim=1)
    gradient_penalty = (GP_STRENGTH / 2.0) * torch.mean(grad_square_sum)

    total_disc_loss = disc_loss + gradient_penalty

    disc.zero_grad()
    total_disc_loss.backward()
    disc_optim.step()

    if step % n_steps_for_summary == 0 and disc_step == 0:
        with torch.no_grad():
            fake = gen(fixed_noise, fixed_labels)
            img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
            img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

            summ_writer.add_image(
                'RotMNIST Fake Images', img_grid_fake, global_step=step
            )
            summ_writer.add_image(
                'RotMNIST Real Images', img_grid_real, global_step=step
            )
            summ_writer.add_scalar(
                'Total Discriminator Loss', total_disc_loss, global_step=step
            )
            summ_writer.add_scalar(
                'Gradient Penalty', gradient_penalty, global_step=step
            )


# ---------------------------------------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------------------------------------
print('\n' + '-' * 32)
print(f'Start training for {EPOCHS} epochs')
print('-' * 32 + '\n')

start_time = time()

for epoch in range(EPOCHS):
    progbar = tqdm(data_loader)
    progbar.set_description(f'EPOCH [{epoch + 1} / {EPOCHS}]')

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
                               step=(i + epoch * steps_per_epoch),
                               disc_step=j
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

    progbar.close()

total_training_time = time() - start_time
hours, remainder = divmod(int(total_training_time), 3600)
minutes, seconds = divmod(remainder, 60)
print(f'Total training time: {hours:02d}h {minutes:02d}m {seconds:02d}s')

# ---------------------------------------------------------------------------------------------------------
# Train loop finished -> save models
# ---------------------------------------------------------------------------------------------------------

print('\n' + '-' * 32)
print(f'Saving Generator and Discriminator as {IDENTIFIER_FOR_SAVING}')
print('-' * 32 + '\n')

# path for trained models
trained_models_path = f'trained_models/{GEN_ARCH}/{current_date}'
# check data directory
if not os.path.isdir(trained_models_path):
    os.mkdir(path=trained_models_path)
    print(f'created new directory {trained_models_path}')

# save models
gen_path = f'{trained_models_path}/generator_{IDENTIFIER_FOR_SAVING}'
disc_path = f'{trained_models_path}/discriminator_{IDENTIFIER_FOR_SAVING}'
torch.save(gen.state_dict(), gen_path)
torch.save(disc.state_dict(), disc_path)