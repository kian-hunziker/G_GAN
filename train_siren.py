import datetime
import os
from sys import platform

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from patchify import unpatchify

from siren import Siren
from utils.lodopab_dataset import PatchedImage
from utils.checkpoints import load_glow_from_checkpoint
from utils.get_device import get_device
from utils.patcher import unpatch
from utils.siren_utils import reshape_z_for_glow
from utils.snr import snr

import warnings


warnings.simplefilter("ignore", UserWarning)

description = 'unnormalized baseline for debugging'


# fix random seeds
torch.manual_seed(0)
np.random.seed(0)

#glow_path = 'trained_models/glow/2023-11-30_13:26:30/checkpoint_100000'
# unnormalized, trained on one image
#glow_path = 'trained_models/glow/2023-12-01_09:01:05/checkpoint_12307'

# normalized, trained on one image
glow_path = 'trained_models/glow/2023-12-28_14:15:34/checkpoint_20000'
img_path = 'datasets/LoDoPaB/ground_truth_train/ground_truth_train_000.hdf5'

img_idx = 0

debug = platform == 'darwin'
device = get_device(debug)

use_grid_sample = True
normalize = True
noise_strength = 0.0

batch_size = 2048
#TODO reduce batch size to avoid NaN
lr = 1e-5
l2_lambda = 0.000
epochs = 300
N = 362
P = 8
patch_dim = N - P + 1
first_omega_0 = 30
hidden_features = 512
hidden_layers = 3

# Setup data loader
train_dataset = PatchedImage(img_path=img_path,
                             img_idx=img_idx,
                             patch_size=P,
                             noise_strength=noise_strength,
                             use_grid_sample=use_grid_sample,
                             normalize=normalize)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_iter = iter(train_loader)

# load GLOW model
print(f'\nLoading GLOW model: \n')
print(f'path: {glow_path}')
glow_model = load_glow_from_checkpoint(glow_path, device=device, arch='lodopab')

# initialize SIREN and optimizer
siren = Siren(in_features=2,
              out_features=64,
              hidden_features=hidden_features,
              hidden_layers=hidden_layers,
              outermost_linear=True,
              first_omega_0=first_omega_0).to(device)
optim = torch.optim.Adam(params=siren.parameters(), lr=lr)
criterion = F.mse_loss

total_iterations = epochs * len(train_dataset) // batch_size

# ---------------------------------------------------------------------------------------------------------
# Setup for summary writer and checkpointing
# ---------------------------------------------------------------------------------------------------------
summary_batch_size = 512
gt_dataset = PatchedImage(img_path=img_path,
                          img_idx=img_idx,
                          patch_size=P,
                          noise_strength=0,
                          use_grid_sample=False,
                          normalize=normalize)
gt_loader = DataLoader(gt_dataset, batch_size=summary_batch_size, shuffle=False)
gt_iter = iter(gt_loader)

losses = []
step_for_summary_loss = 5
step_for_summary_reconstruction = 100
step_for_checkpoint = 1000

current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
log_dir = f'runs/siren/{current_date}'
summ_writer = SummaryWriter(log_dir)

training_image = unpatch(train_dataset.get_all_coords_and_patches()[-1])
clean_image = unpatch(gt_dataset.get_all_coords_and_patches()[-1])

summ_writer.add_image(
    'Training Image', training_image, global_step=0, dataformats='HW'
)
summ_writer.add_image(
    'Clean Image', clean_image, global_step=0, dataformats='HW'
)

initial_snr = snr(
    torch.from_numpy(clean_image).unsqueeze(0).unsqueeze(0),
    torch.from_numpy(training_image).unsqueeze(0).unsqueeze(0)
)

# setup for checkpointing and saving trained models
trained_models_path = f'trained_models/siren'
if not os.path.isdir(trained_models_path):
    os.mkdir(path=trained_models_path)
    print(f'created new directory {trained_models_path}')
trained_models_path = f'{trained_models_path}/{current_date}'
if not os.path.isdir(trained_models_path):
    os.mkdir(path=trained_models_path)
    print(f'created new directory {trained_models_path}')


def save_checkpoint(n_iterations, loss_hist):
    checkpoint_path = f'{trained_models_path}/checkpoint_{n_iterations}'
    checkpoint = {
        'iterations': n_iterations,
        'model_arch': 'siren',
        'model': siren.state_dict(),
        'hidden_features': hidden_features,
        'hidden_layers': hidden_layers,
        'lr': lr,
        'l2_lambda': l2_lambda,
        'batch_size': batch_size,
        'omega_0': first_omega_0,
        'loss_hist': loss_hist,
        'img_idx': img_idx,
        'noise_strength': noise_strength,
        'use_grid_sample': use_grid_sample,
        'description': description,
    }
    torch.save(checkpoint, checkpoint_path)


# ---------------------------------------------------------------------------------------------------------
# Train SIREN
# ---------------------------------------------------------------------------------------------------------
print('-' * 32)
print(f'Start training')
print(f'device: {device}')
print(f'Training on image with idx: {img_idx}')
print(f'learning rate: {lr}')
print(f'L2 lambda: {l2_lambda}')
print(f'omega_0: {first_omega_0}')
print(f'hidden features: {hidden_features}')
print(f'hidden layers: {hidden_layers}')
print(f'batch size: {batch_size}')
print(f'num epochs: {epochs}')
print(f'noise strength: {noise_strength}')
print(f'SNR between clean and training image: {initial_snr.detach().item() :.3f}dB')
print(f'using grid_sample: {use_grid_sample}')
print(f'description: {description}')
print('-' * 32 + '\n')

prog_bar = tqdm(total=total_iterations)

for step in range(total_iterations):
    # get coords in range [-1, 1] and corresponding patches
    try:
        coords, true_patches = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        coords, true_patches = next(train_iter)
    coords = coords.to(device)
    true_patches = true_patches.to(device)

    # pass coords through siren to get proposed latent vector z_siren
    z_siren, _ = siren(coords)

    # reshape latent vector to pass it to glow model
    z = reshape_z_for_glow(z_siren, glow_model)

    # generate patches from z by passing latent vector to generative glow model
    glow_patches, _ = glow_model.forward_and_log_det(z)

    # compute MSE loss
    z_l2_loss = l2_lambda * torch.mean(torch.linalg.norm(z_siren, dim=1) ** 2)

    if torch.sum(torch.isnan(glow_patches)) > 1:
        print(f'Nan values in glow_patches')
        break
    if torch.sum(torch.isnan(true_patches)) > 1:
        print(f'Nan values in true patches')
        break

    mse_loss = criterion(glow_patches.squeeze(), true_patches)

    if torch.sum(torch.isnan(mse_loss)) > 1:
        print(f'Nan values in MSE loss')
        break

    loss = mse_loss + z_l2_loss

    if torch.sum(torch.isnan(loss)) > 1:
        print(f'Nan values in loss')
        break

    #TODO + 0.005 * torch.linalg.norm(z_siren)**2
    # loss = criterion(glow_patches.squeeze()[:, 4:5, 4:5], true_patches[:, 4:5, 4:5]) #+ 0.005 * torch.linalg.norm(z_siren)
    losses.append(loss.detach().cpu().numpy())

    # gradient descent
    optim.zero_grad()
    loss.backward()
    optim.step()

    # summary
    if step % step_for_summary_loss == 0:
        summ_writer.add_scalar(
            'Loss', loss, global_step=step
        )
        summ_writer.add_scalar(
            'MSE loss', mse_loss, global_step=step
        )
        summ_writer.add_scalar(
            'L2 loss z', z_l2_loss, global_step=step
        )
    if not debug and step % step_for_summary_reconstruction == 0:
        summary_patches = []
        gt_iter = iter(gt_loader)
        with torch.no_grad():
            for i, (gt_coords, gt_patches) in enumerate(gt_iter):
                z_summary, _ = siren(gt_coords.to(device))
                z_summary = reshape_z_for_glow(z_summary, glow_model)
                temp_patches, _ = glow_model.forward_and_log_det(z_summary)
                summary_patches.append(temp_patches.detach().cpu())
        summary_patches = torch.cat(summary_patches).numpy().reshape(patch_dim, patch_dim, P, P)
        # unpatchify does not use averaging
        #summary_reconstruction = unpatchify(summary_patches, (N, N))

        # use averaging to reconstruct image from overlapping patches
        summary_reconstruction = unpatch(summary_patches, stride=1)

        with torch.no_grad():
            reconstruction_loss = F.mse_loss(torch.from_numpy(training_image), torch.from_numpy(summary_reconstruction))
            target = torch.from_numpy(clean_image).unsqueeze(0).unsqueeze(0)
            approx = torch.from_numpy(summary_reconstruction).unsqueeze(0).unsqueeze(0)
            current_snr = snr(target, approx)

        summ_writer.add_scalar(
            'Reconstruction Loss', reconstruction_loss, global_step=step
        )

        summ_writer.add_scalar(
            'SNR', current_snr, global_step=step
        )

        summ_writer.add_image(
            'Reconstruction', summary_reconstruction, global_step=step, dataformats='HW'
        )
    if step > 0 and step % step_for_checkpoint == 0:
        save_checkpoint(step, losses)

    prog_bar.update(1)

prog_bar.close()
save_checkpoint(step, losses)
