import datetime
import os
from sys import platform

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import h5py
from patchify import patchify, unpatchify
from math import prod

from siren import Siren
from utils.siren_mgrid import get_mgrid
from utils.checkpoints import load_glow_from_checkpoint
from utils.get_device import get_device
from utils.patcher import unpatch

import warnings

warnings.simplefilter("ignore", UserWarning)


class PatchedImage(Dataset):
    def __init__(self, img_path: str, img_idx: int = 0, patch_size: int = 8):
        super().__init__()
        # load single image
        f = h5py.File(img_path)
        img = f['data'][img_idx]

        # generate patches
        self.patches = patchify(img, patch_size=(patch_size, patch_size), step=1)

        # generate 2D coords in range [-1, 1]
        self.coords = get_mgrid(self.patches.shape[0], dim=2)
        # reverse order of y-coords. The top left corner should have coords [-1, 1]
        temp_x = 1.0 * self.coords[:, 0]
        self.coords[:, 0] = 1.0 * self.coords[:, 1]
        self.coords[:, 1] = 1.0 * temp_x
        self.coords[:, 1] = -1.0 * self.coords[:, 1]

    def __len__(self):
        return self.patches.shape[0] * self.patches.shape[1]

    def __getitem__(self, idx):
        x_idx = idx // self.patches.shape[0]
        y_idx = idx % self.patches.shape[1]
        return self.coords[idx], self.patches[x_idx, y_idx]


def reshape_z_for_glow(z_vec, glow_instance):
    z = []
    start = 0
    curr_batch_size = z_vec.shape[0]
    for q in glow_instance.q0:
        length = int(prod(q.shape))
        z_temp = z_vec[:, start:start + length]
        start += length
        z_temp = z_temp.reshape((curr_batch_size,) + q.shape)
        z.append(z_temp)
    return z


description = 'Glow single image, 4 hidden layers'


# fix random seeds
torch.manual_seed(0)
np.random.seed(0)

#glow_path = 'trained_models/glow/2023-11-30_13:26:30/checkpoint_100000'
glow_path = 'trained_models/glow/2023-12-01_09:01:05/checkpoint_12307'
img_path = 'datasets/LoDoPaB/ground_truth_train/ground_truth_train_000.hdf5'

img_idx = 1

debug = platform == 'darwin'
device = get_device(debug)

batch_size = 2048
lr = 1e-5
epochs = 300
N = 362
P = 8
patch_dim = N - P + 1
first_omega_0 = 30
hidden_features = 512
hidden_layers = 4

# Setup data loader
patched_dataset = PatchedImage(img_path=img_path, img_idx=img_idx, patch_size=P)
patched_loader = DataLoader(patched_dataset, batch_size=batch_size, shuffle=True)
patched_iter = iter(patched_loader)

# load GLOW model
print(f'\nLoading GLOW model: \n')
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

total_iterations = epochs * len(patched_dataset) // batch_size

# ---------------------------------------------------------------------------------------------------------
# Setup for summary writer and checkpointing
# ---------------------------------------------------------------------------------------------------------
summary_batch_size = 512
gt_loader = DataLoader(patched_dataset, batch_size=summary_batch_size, shuffle=False)
gt_iter = iter(gt_loader)

losses = []
step_for_summary_loss = 5
step_for_summary_reconstruction = 100
step_for_checkpoint = 1000

current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
log_dir = f'runs/siren/{current_date}'
summ_writer = SummaryWriter(log_dir)

all_patches = []
gt_iter = iter(gt_loader)

# add ground truth image to summary writer
for i, (gt_coords, gt_patches) in enumerate(gt_iter):
    all_patches.append(gt_patches)

all_patches = torch.cat(all_patches)
gt_image = unpatchify(all_patches.numpy().reshape(patch_dim, patch_dim, P, P), (N, N))
summ_writer.add_image(
    'Ground Truth', gt_image, global_step=0, dataformats='HW'
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
        'batch_size': batch_size,
        'omega_0': first_omega_0,
        'loss_hist': loss_hist,
        'img_idx': img_idx,
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
print(f'omega_0: {first_omega_0}')
print(f'hidden features: {hidden_features}')
print(f'hidden layers: {hidden_layers}')
print(f'batch size: {batch_size}')
print(f'num epochs: {epochs}')
print(f'description: {description}')
print('-' * 32 + '\n')

prog_bar = tqdm(total=total_iterations)

for step in range(total_iterations):
    # get coords in range [-1, 1] and corresponding patches
    try:
        coords, true_patches = next(patched_iter)
    except StopIteration:
        patched_iter = iter(patched_loader)
        coords, true_patches = next(patched_iter)
    coords = coords.to(device)
    true_patches = true_patches.to(device)

    # pass coords through siren to get proposed latent vector z_siren
    z_siren, _ = siren(coords)

    # reshape latent vector to pass it to glow model
    z = reshape_z_for_glow(z_siren, glow_model)

    # generate patches from z by passing latent vector to generative glow model
    glow_patches, _ = glow_model.forward_and_log_det(z)

    # compute MSE loss
    loss = criterion(glow_patches.squeeze(), true_patches)
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
            reconstruction_loss = F.mse_loss(torch.from_numpy(gt_image), torch.from_numpy(summary_reconstruction))

        summ_writer.add_scalar(
            'Reconstruction Loss', reconstruction_loss, global_step=step
        )

        summ_writer.add_image(
            'Reconstruction', summary_reconstruction, global_step=step, dataformats='HW'
        )
    if step > 0 and step % step_for_checkpoint == 0:
        save_checkpoint(step, losses)

    prog_bar.update(1)

prog_bar.close()
save_checkpoint(step, losses)
