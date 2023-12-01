import os

import torch
import torchvision
import normflows as nf
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

import datetime
from tqdm import tqdm

import glow_models
import utils.lodopab_dataset
from utils.data_loaders import get_rotated_mnist_dataloader, get_standard_mnist_dataloader
from utils.get_device import get_device

import warnings
warnings.simplefilter("ignore", UserWarning)

torch.manual_seed(0)

debug = False
device = get_device(debug)
project_root = os.getcwd()

# ---------------------------------------------------------------------------------------------------------
# Set up model
# ---------------------------------------------------------------------------------------------------------
LR = 1e-3
WD = 1e-5


model = glow_models.get_lodopab_glow_model()
model = model.to(device)

# ---------------------------------------------------------------------------------------------------------
# Prepare training data
# ---------------------------------------------------------------------------------------------------------
'''
batch_size = 128

dataset, train_loader = get_rotated_mnist_dataloader(root=project_root,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     one_hot_encode=False,
                                                     num_examples=60000,
                                                     num_rotations=0,
                                                     no_labels=False,
                                                     img_size=image_size,
                                                     single_class=None,
                                                     glow=True)
                                                     '''

lodopad_path = 'datasets/LoDoPaB/ground_truth_train/ground_truth_train_000.hdf5'
batch_size = 2048
num_images = 1
dataset = utils.lodopab_dataset.LodopabDataset(file_path=lodopad_path, patch_size=8, num_images=num_images)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

train_iter = iter(train_loader)

# ---------------------------------------------------------------------------------------------------------
# Setup for summary writer and checkpointing
# ---------------------------------------------------------------------------------------------------------
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
log_dir = f'runs/glow/{current_date}'
summ_writer = SummaryWriter(log_dir)

n_summary_examples = 40
n_steps_for_summary = 500
n_steps_for_checkpoint = 10000

# setup for checkpointing and saving trained models
trained_models_path = f'trained_models/glow'
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
        'gen_arch': 'glow',
        'disc_arch': '',
        'generator': model.state_dict(),
        'lr': LR,
        'wd': WD,
        'num_images': num_images,
        'loss_hist': loss_hist,
    }
    torch.save(checkpoint, checkpoint_path)


# ---------------------------------------------------------------------------------------------------------
# Train model
# ---------------------------------------------------------------------------------------------------------

epochs = 200
max_iter = int(epochs * len(dataset) / batch_size)
step = 0

print('\n' + '-' * 32)
print(f'Start training GLOW for {max_iter} iterations')
print('-' * 32 + '\n')
print(f'Device: {device}')
print(f'Batch size: {batch_size}')
print(f'LR: {LR}')
print(f'WD: {WD}')
print(f'Number of training images: {num_images}')
print('\n')


loss_hist = np.array([])

optimizer = torch.optim.Adamax(model.parameters(), lr=LR, weight_decay=WD)

for i in tqdm(range(max_iter)):
    try:
        x = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x = next(train_iter)
    optimizer.zero_grad()
    loss = model.forward_kld(x.to(device))

    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

    loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())

    if (step < 1000 and step % 10 == 0) or (step % n_steps_for_summary == 0):
        with torch.no_grad():
            fake, _ = model.sample(num_samples=n_summary_examples, y=None)
            #fake_ = torch.clamp(fake, 0, 1)
            img_grid_fake = torchvision.utils.make_grid(fake, nrow=10)
            img_grid_real = torchvision.utils.make_grid(x[:32])

            summ_writer.add_image(
                'Fake Patches', img_grid_fake, global_step=step
            )
            summ_writer.add_image(
                'Real Patches', img_grid_real, global_step=step
            )
            summ_writer.add_scalar(
                'Model Loss', loss, global_step=step
            )

    if step > 0 and step % n_steps_for_checkpoint == 0:
        save_checkpoint(step, loss_hist)

    step += 1

save_checkpoint(step, loss_hist)

