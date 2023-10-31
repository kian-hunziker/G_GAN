import os

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import random
import datetime
from time import time
from tqdm import tqdm

from generators import Generator, initialize_weights
from discriminators import Discriminator
from utils.dataLoaders import get_rotated_mnist_dataloader, get_standard_mnist_dataloader
from utils.optimizers import get_optimizers
from utils.gradient_penalty import zero_centered_gp_real_data, vanilla_gp

# TODO name networks properly
IDENTIFIER_FOR_SAVING = 'test'

# setup device
device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')

# Root path of project
project_root = os.getcwd()

print('-' * 32 + '\n')
print(f'Using device: {device}')
print(f'Project root: {project_root}\n')

# Hyperparameters
USE_GGAN_TRAINING_LOOP = False

N_ITER_FOR_CHECKPOINT = 1000
N_STEPS_FOR_SUMMARY = 10

# GEN_ARCH: one of {'z2_rot_mnist_no_label', 'z2_rot_mnist', 'p4_rot_mnist', 'vanilla', 'vanilla_small'}
GEN_ARCH = 'vanilla_small'
# DISC_ARCH: one of {'z2_rot_mnist_no_label', 'z2_rot_mnist', 'p4_rot_mnist', 'vanilla', 'vanilla_small'}
DISC_ARCH = 'vanilla_small'

DISC_UPDATE_STEPS = 5
# LOSS_TYPE: one of {'wasserstein', 'rel_avg'}
LOSS_TYPE = 'wasserstein'
# GP_TYPE: one of {'vanilla', 'zero_centered'}
GP_TYPE = 'vanilla'
GP_STRENGTH = 0.1  # 10 or 0.1

EPOCHS = 300
BATCH_SIZE = 64
NUM_CLASSES = 10
LATENT_DIM = 64
EPS = 1e-6

IMG_SHAPE = (1, 28, 28)

BETA_1 = 0.0
BETA_2 = 0.9
LR_G = 0.0001
LR_D = 0.0001  # 0.0004

# fix random seeds
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------------------------------------
# Initialize models, loaders and optimizers
# ---------------------------------------------------------------------------------------------------------
dataset, data_loader = get_rotated_mnist_dataloader(root=project_root,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    one_hot_encode=True,
                                                    num_examples=60000,
                                                    num_rotations=0,
                                                    no_labels='no_label' in GEN_ARCH and 'no_label' in DISC_ARCH,
                                                    img_size=28)
'''
dataset, data_loader = get_standard_mnist_dataloader(root=project_root,
                                                     img_size=28)
'''
print(f'Total number of training examples: {len(dataset)}')

# setup generator and discriminator
gen = Generator(n_classes=NUM_CLASSES, gen_arch=GEN_ARCH, latent_dim=LATENT_DIM)
disc = Discriminator(img_shape=IMG_SHAPE, disc_arch=DISC_ARCH, n_classes=NUM_CLASSES)

n_trainable_params_gen = sum(p.numel() for p in gen.parameters() if p.requires_grad)
n_trainable_params_disc = sum(p.numel() for p in disc.parameters() if p.requires_grad)
print(f'Trainable parameters in Generator: {n_trainable_params_gen}')
print(f'Trainable parameters in Discriminator: {n_trainable_params_disc}\n')

# weight initialization
print('Initializing Generator Weights')
initialize_weights(gen, gen.gen_arch)
print('Initializing Discriminator Weights')
initialize_weights(disc, disc.disc_arch)

gen = gen.to(device)
disc = disc.to(device)
gen.train()
disc.train()

# get optimizers
gen_optim, disc_optim = get_optimizers(lr_g=LR_G,
                                       lr_d=LR_D,
                                       beta1_g=BETA_1,
                                       beta2_g=BETA_2,
                                       beta1_d=BETA_1,
                                       beta2_d=BETA_2,
                                       gen=gen,
                                       disc=disc)


# ---------------------------------------------------------------------------------------------------------
# Setup for summary writer and checkpointing
# ---------------------------------------------------------------------------------------------------------
current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_dir = f'runs/{GEN_ARCH}/{current_date}'
summ_writer = SummaryWriter(log_dir)

summ_writer.add_text('gen_arch', GEN_ARCH)
summ_writer.add_text('disc_arch', DISC_ARCH)
summ_writer.add_text('loss function', LOSS_TYPE)
summ_writer.add_text('GP type', GP_TYPE)
summ_writer.add_text('GP strength', f'{GP_STRENGTH}')
summ_writer.add_text('lr', f'lr generator: {LR_G}, lr discriminator: {LR_D}')
summ_writer.add_text('disc update steps', f'Disc update steps: {DISC_UPDATE_STEPS}')

# create fixed latent noise
no_examples_for_summary = 40
fixed_noise = torch.randn(no_examples_for_summary, LATENT_DIM).to(device)

# create fixed labels
l = torch.zeros(no_examples_for_summary, 10)
for i in range(4):
    for c in range(10):
        l[10 * i + c, c] = 1

fixed_labels = l.to(device)
# fixed_labels = torch.zeros(fixed_labels.shape)
# fixed_labels[:, 1] = torch.ones(no_examples_for_summary)
# fixed_labels = fixed_labels.to(device)
if 'no_label' in GEN_ARCH:
    fixed_labels = None
print(f'\nThe fixed labels are: \n {fixed_labels.nonzero(as_tuple=True)[1].view(4, 10).cpu().numpy()}')

# setup for checkpointing and saving trained models
trained_models_path = f'trained_models/{GEN_ARCH}'
if not os.path.isdir(trained_models_path):
    os.mkdir(path=trained_models_path)
    print(f'created new directory {trained_models_path}')
trained_models_path = f'{trained_models_path}/{current_date}'
if not os.path.isdir(trained_models_path):
    os.mkdir(path=trained_models_path)
    print(f'created new directory {trained_models_path}')


def save_checkpoint(n_iterations):
    checkpoint_path = f'{trained_models_path}/checkpoint_{n_iterations}'
    torch.save({
        'iterations': n_iterations,
        'gen_arch': GEN_ARCH,
        'disc_arch': DISC_ARCH,
        'generator': gen.state_dict(),
        'discriminator': disc.state_dict(),
        'gen_optim': gen_optim.state_dict(),
        'disc_optim': disc_optim.state_dict(),
    }, checkpoint_path)


# ---------------------------------------------------------------------------------------------------------
# Training steps for generator and discriminator
# ---------------------------------------------------------------------------------------------------------
def gen_training_step(real_batch, labels, noise_batch, step, eps=EPS):
    fake_batch = gen(noise_batch, labels)

    if LOSS_TYPE == 'rel_avg':
        disc_opinion_real = disc([real_batch, labels])
        disc_opinion_fake = disc([fake_batch, labels])

        real_fake_rel_avg_opinion = (disc_opinion_real - torch.mean(disc_opinion_fake, dim=0))
        fake_real_rel_avg_opinion = (disc_opinion_fake - torch.mean(disc_opinion_real, dim=0))

        # loss, relativistic average loss
        gen_loss = torch.mean(
            - torch.mean(torch.log(torch.sigmoid(fake_real_rel_avg_opinion) + eps), dim=0)
            - torch.mean(torch.log(1 - torch.sigmoid(real_fake_rel_avg_opinion) + eps), dim=0)
        )
    elif LOSS_TYPE == 'wasserstein':
        disc_opinion_fake = disc([fake_batch, labels]).reshape(-1)
        gen_loss = -torch.mean(disc_opinion_fake)

    gen.zero_grad()
    gen_loss.backward()
    gen_optim.step()

    if step % N_STEPS_FOR_SUMMARY == 0:
        summ_writer.add_scalar(
            'Generator Loss', gen_loss, global_step=step
        )


def disc_training_step(real_batch, labels, noise_batch, step, disc_step, eps=EPS):
    # TODO maybe generate fake images without gradient??
    with torch.no_grad():
        fake_batch = gen(noise_batch, labels)

    disc_opinion_real = disc([real_batch, labels])
    disc_opinion_fake = disc([fake_batch, labels])

    if LOSS_TYPE == 'rel_avg':
        real_fake_rel_avg_opinion = (disc_opinion_real - torch.mean(disc_opinion_fake, dim=0))
        fake_real_rel_avg_opinion = (disc_opinion_fake - torch.mean(disc_opinion_real, dim=0))

        disc_loss = torch.mean(
            - torch.mean(torch.log(torch.sigmoid(real_fake_rel_avg_opinion) + eps), dim=0)
            - torch.mean(torch.log(1 - torch.sigmoid(fake_real_rel_avg_opinion) + eps), dim=0)
        )
    elif LOSS_TYPE == 'wasserstein':
        disc_loss = (-(torch.mean(disc_opinion_real.reshape(-1)) - torch.mean(disc_opinion_fake.reshape(-1))))

    # TODO check Gradient Penalty
    if GP_TYPE == 'zero_centered':
        gradient_penalty = zero_centered_gp_real_data(disc, real_batch, labels, device)
    elif GP_TYPE == 'vanilla':
        gradient_penalty = vanilla_gp(disc, real_batch, fake_batch, device)
    elif GP_TYPE == 'no_gp':
        gradient_penalty = 0.0

    total_disc_loss = disc_loss + GP_STRENGTH * gradient_penalty

    disc.zero_grad()
    # TODO: double check retain_graph=True for gGAN architectures
    total_disc_loss.backward(retain_graph=True)
    disc_optim.step()

    if step % N_STEPS_FOR_SUMMARY == 0 and disc_step == 0:
        # write summary entry with real and fake images
        with torch.no_grad():
            fake = gen(fixed_noise, fixed_labels)
            img_grid_fake = torchvision.utils.make_grid(fake[:no_examples_for_summary], nrow=10, normalize=True)
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
# Train loops
# ---------------------------------------------------------------------------------------------------------
print('\n' + '-' * 32)
print(f'Start training for {EPOCHS} epochs')
print('-' * 32 + '\n')

no_training_examples = len(dataset)
start_time = time()

if USE_GGAN_TRAINING_LOOP is False:
    print(f'Using DC_GAN training loop with {DISC_UPDATE_STEPS} disc updates per generator update')
    print(f'Training disc {DISC_UPDATE_STEPS} times and gen once on a single batch')
    steps_per_epoch = int(no_training_examples // BATCH_SIZE)

    for epoch in range(EPOCHS):
        progbar = tqdm(data_loader)
        progbar.set_description(f'EPOCH [{epoch + 1} / {EPOCHS}]')

        # loop through dataset
        for i in range(steps_per_epoch):
            step = i + epoch * steps_per_epoch
            real, labels = next(iter(data_loader))
            real = real.to(device)
            labels = labels.to(device)

            # update discriminator
            for j in range(DISC_UPDATE_STEPS):
                noise = torch.randn(real.shape[0], LATENT_DIM).to(device)

                disc_training_step(real_batch=real,
                                   labels=labels,
                                   noise_batch=noise,
                                   step=step,
                                   disc_step=j
                                   )

            # update generator
            noise = torch.randn(real.shape[0], LATENT_DIM).to(device)

            gen_training_step(real_batch=real,
                              labels=labels,
                              noise_batch=noise,
                              step=step
                              )

            if step % N_ITER_FOR_CHECKPOINT == 0:
                save_checkpoint(step)

            progbar.update(1)

        progbar.close()

else:
    print(f'Using group equiv GAN training loop with {DISC_UPDATE_STEPS} disc updates per generator update')
    print(f'Loading new real batch for every update step\n')
    examples_per_iteration = BATCH_SIZE * (DISC_UPDATE_STEPS + 1)
    steps_per_epoch = int(no_training_examples // examples_per_iteration)

    for epoch in range(EPOCHS):
        progbar = tqdm(data_loader)
        progbar.set_description(f'EPOCH [{epoch + 1} / {EPOCHS}]')

        # loop through dataset
        for i in range(steps_per_epoch):

            step = i + epoch * steps_per_epoch
            # update discriminator
            for j in range(DISC_UPDATE_STEPS):
                real, labels = next(iter(data_loader))
                noise = torch.randn((real.shape[0], LATENT_DIM), device=device)
                real = real.to(device)
                labels = labels.to(device)

                disc_training_step(real_batch=real,
                                   labels=labels,
                                   noise_batch=noise,
                                   step=step,
                                   disc_step=j
                                   )

            # update generator
            real, labels = next(iter(data_loader))
            noise = torch.randn((real.shape[0], LATENT_DIM), device=device)
            real = real.to(device)
            labels = labels.to(device)

            gen_training_step(real_batch=real,
                              labels=labels,
                              noise_batch=noise,
                              step=step
                              )

            if step % N_ITER_FOR_CHECKPOINT == 0:
                save_checkpoint(step)

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

# save models
gen_path = f'{trained_models_path}/generator_{IDENTIFIER_FOR_SAVING}'
disc_path = f'{trained_models_path}/discriminator_{IDENTIFIER_FOR_SAVING}'
torch.save(gen.state_dict(), gen_path)
torch.save(disc.state_dict(), disc_path)
