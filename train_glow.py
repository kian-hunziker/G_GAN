import os

import torch
import torchvision
import normflows as nf
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import datetime
from tqdm import tqdm

from utils.dataLoaders import get_rotated_mnist_dataloader, get_standard_mnist_dataloader
from utils.getDevice import get_device

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

# Define flows
L = 3
K = 16

image_size = 16
input_shape = (1, image_size, image_size)
n_dims = np.prod(input_shape)
channels = 1
hidden_channels = 256
split_mode = 'channel'
scale = True
num_classes = 10

# Set up flows, distributions and merge operations
q0 = []
merges = []
flows = []
for i in range(L):
    flows_ = []
    for j in range(K):
        flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                      split_mode=split_mode, scale=scale)]
    flows_ += [nf.flows.Squeeze()]
    flows += [flows_]
    if i > 0:
        merges += [nf.flows.Merge()]
        latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i),
                        input_shape[2] // 2 ** (L - i))
    else:
        latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L,
                        input_shape[2] // 2 ** L)
    q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]

# Construct flow model with the multiscale architecture
model = nf.MultiscaleFlow(q0, flows, merges)
model = model.to(device)

# ---------------------------------------------------------------------------------------------------------
# Prepare training data
# ---------------------------------------------------------------------------------------------------------

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

train_iter = iter(train_loader)

# ---------------------------------------------------------------------------------------------------------
# Setup for summary writer and checkpointing
# ---------------------------------------------------------------------------------------------------------
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
log_dir = f'runs/glow/{current_date}'
summ_writer = SummaryWriter(log_dir)

n_summary_examples = 4
n_steps_for_summary = 500
n_steps_for_checkpoint = 10000
summary_labels = torch.arange(num_classes).repeat(n_summary_examples).to(device)

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
        'loss_hist': loss_hist
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
print(f'Img size: {image_size}')
print('\n')


loss_hist = np.array([])

optimizer = torch.optim.Adamax(model.parameters(), lr=LR, weight_decay=WD)

for i in tqdm(range(max_iter)):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
    optimizer.zero_grad()
    loss = model.forward_kld(x.to(device), y.to(device))

    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

    loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())

    if (step < 1000 and step % 10 == 0) or (step % n_steps_for_summary == 0):
        with torch.no_grad():
            fake, _ = model.sample(y=summary_labels)
            fake_ = torch.clamp(fake, 0, 1)
            img_grid_fake = torchvision.utils.make_grid(fake_, nrow=10)
            img_grid_real = torchvision.utils.make_grid(x[:32], normalize=True)

            summ_writer.add_image(
                'RotMNIST Fake Images', img_grid_fake, global_step=step
            )
            summ_writer.add_image(
                'RotMNIST Real Images', img_grid_real, global_step=step
            )
            summ_writer.add_scalar(
                'Model Loss', loss, global_step=step
            )

    if step % n_steps_for_checkpoint == 0:
        save_checkpoint(step, loss_hist)

    step += 1

save_checkpoint(step, loss_hist)

