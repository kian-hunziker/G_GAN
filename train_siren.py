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

description = 'aggregate batches'

# fix random seeds
torch.manual_seed(0)
np.random.seed(0)

# glow_path = 'trained_models/glow/2023-11-30_13:26:30/checkpoint_100000'
# unnormalized, trained on one image
# glow_path = 'trained_models/glow/2023-12-01_09:01:05/checkpoint_12307'

# normalized, trained on one image
glow_path = 'trained_models/glow/2023-12-28_14:15:34/checkpoint_246142'
img_path = 'datasets/LoDoPaB/ground_truth_train/ground_truth_train_000.hdf5'

img_idx = 0

debug = platform == 'darwin'
device = get_device(debug)

use_grid_sample = True
normalize = True
use_averaged_pixel_values = True
noise_strength = 0.0

batch_size = 32
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
                             patch_size=P if use_averaged_pixel_values is False else 1,
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


pixel_offset = 2.0 / N / train_dataset.coord_range
k = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]


def get_coords_average_patching(coordinates: torch.Tensor) -> torch.Tensor:
    """
    For each (x, y) coordinate in coords, we create coordinates for 64 overlapping patches. They each
    overlap by one pixel.
    :param coordinates: Tensor containing 2D coordinates between [-1, 1] for both x and y. Dim: [batch_size, 2]
    :return: Expanded coordinates. dim: [64 * batch_size, 2]
    """
    all_coordinates = []
    for i in range(coordinates.shape[0]):
        coord_list = []
        for k_1 in k:
            for k_2 in k:
                c = coordinates[i] + pixel_offset * torch.tensor([k_2, -k_1])
                coord_list.append(c)
        all_coordinates.append(torch.stack(coord_list))
    return torch.cat(all_coordinates)


def compute_averaged_pixel_value(patches: torch.Tensor):
    patches = patches.squeeze()
    x_range = [7, 6, 5, 4, 3, 2, 1, 0]
    num_pixels = int(patches.shape[0] / 64)
    averaged_pixels = torch.zeros(num_pixels)

    idx = 0
    for i in range(num_pixels):
        pix = 0.0
        for col in x_range:
            for row in x_range:
                inc = patches[idx][col, row]  # .item()
                pix += inc
                idx += 1
        averaged_pixels[i] = pix / 64.0
    return averaged_pixels


# ---------------------------------------------------------------------------------------------------------
# Train SIREN
# ---------------------------------------------------------------------------------------------------------
print('-' * 32)
print(f'Start training')
print(f'device: {device}')
print(f'Training on image with idx: {img_idx}')
print(f'Using averaged pixel values: {use_averaged_pixel_values}')
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

zero_train_iterations = 200
zero_optim = torch.optim.Adam(params=siren.parameters(), lr=1e-4)
for i in range(zero_train_iterations):
    coords = 2.0 * (torch.rand((64, 2)) - 0.5)
    coords = coords.to(device)
    out, _ = siren(coords)
    gt = torch.zeros_like(out)
    loss = criterion(out, gt)
    zero_optim.zero_grad()
    loss.backward()
    zero_optim.step()

print('\n')
print(f'final loss after pretraining: {loss.detach().item()}')
print('\n')

prog_bar = tqdm(total=total_iterations)
pix_values = []
true_pixels = []
z2_losses = []

for step in range(total_iterations):
    # get coords in range [-1, 1] and corresponding patches
    try:
        coords, true_patches = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        coords, true_patches = next(train_iter)

    if use_averaged_pixel_values is True:
        coords = get_coords_average_patching(coords).to(device)
    else:
        coords = coords.to(device)

    true_patches = true_patches.to(device)

    # pass coords through siren to get proposed latent vector z_siren
    z_siren, _ = siren(coords)

    # reshape latent vector to pass it to glow model
    z = reshape_z_for_glow(z_siren, glow_model)

    # generate patches from z by passing latent vector to generative glow model
    glow_patches, _ = glow_model.forward_and_log_det(z)

    if torch.sum(torch.isnan(glow_patches)) > 1:
        print(f'Nan values in glow_patches')
        break
    if torch.sum(torch.isnan(true_patches)) > 1:
        print(f'Nan values in true patches')
        break

    # compute loss
    if use_averaged_pixel_values is True:
        pixel_values = compute_averaged_pixel_value(glow_patches).to(device)
        pix_values.append(pixel_values)
        true_pixels.append(true_patches)
        z_l2_loss = l2_lambda * torch.mean(torch.linalg.norm(z_siren, dim=1) ** 2)
        z2_losses.append(z_l2_loss)
        with torch.no_grad():
            mse_loss = criterion(pixel_values, true_patches)
        if step > 0 and step % 4 == 0:
            pix_values = torch.cat(pix_values)
            true_pixels = torch.cat(true_pixels)
            z_l2_loss = torch.mean(torch.stack(z2_losses))

            mse_loss = criterion(pix_values, true_pixels)
            loss = mse_loss + z_l2_loss

            # gradient descent
            optim.zero_grad()
            loss.backward()
            optim.step()

            pix_values = []
            true_pixels = []
            z2_losses = []
    else:
        z_l2_loss = l2_lambda * torch.mean(torch.linalg.norm(z_siren, dim=1) ** 2)
        mse_loss = criterion(glow_patches.squeeze(), true_patches)
        loss = mse_loss + z_l2_loss
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
        # summary_reconstruction = unpatchify(summary_patches, (N, N))

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
