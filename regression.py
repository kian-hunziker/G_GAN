import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

import random
import datetime
from time import time
from tqdm import tqdm

from generators import Generator, init_generator_weights_z2
from discriminators import Discriminator, init_discriminator_weights_z2
from utils.dataLoaders import get_rotated_mnist_dataloader
from utils.checkpoints import load_checkpoint

device = 'cpu'

lr = 1e-4
class_to_search = 8
n_iterations = 2000
n_training_examples = 12000

step_for_plot = 100

#gen, _ = load_checkpoint('trained_models/p4_rot_mnist/2023-10-17 22:02:15/checkpoint_10000', device=device)
gen = Generator()
gen.load_state_dict(
    torch.load('trained_models/z2_rot_mnist/2023-10-13 18:23:50/generator_test',
               map_location=torch.device('cpu'))
)
gen.eval()

latent_noise = torch.zeros(1, 64, requires_grad=True, dtype=torch.float, device=device)
label = torch.zeros(1, 10, device=device)
label[0, class_to_search] = 1

project_root = os.path.dirname(os.path.abspath(__file__))
dataset, _ = get_rotated_mnist_dataloader(root=project_root,
                                          batch_size=64,
                                          shuffle=True,
                                          one_hot_encode=True,
                                          num_examples=60000,
                                          num_rotations=0)

indices_of_target_class = np.where(dataset.targets[n_training_examples:] == class_to_search)[0]
target_idx = indices_of_target_class[random.randint(0, len(indices_of_target_class) - 1)] + n_training_examples
target = dataset.data[target_idx]

plt.imshow(target, cmap='gray')
plt.title('Target')
plt.show()

transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
target = transform(target)

optim = torch.optim.Adam(params=[latent_noise], lr=lr)
criterion = torch.nn.MSELoss(reduction='sum')

losses = []
prog_bar = tqdm(total=n_iterations)

for i in range(n_iterations):
    approx = gen(latent_noise, label)[0]
    loss = criterion(approx, target)

    losses.append(loss.detach().cpu().numpy())

    loss.backward()
    optim.step()

    if i % step_for_plot == 0:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(target.detach().cpu().numpy()[0], cmap='gray')
        ax[0].set_title('Target')
        ax[1].imshow(approx.detach().cpu().numpy()[0], cmap='gray')
        ax[1].set_title('Approximation')
        plt.suptitle(f'Iteration {i} / {n_iterations}')
        plt.tight_layout()
        plt.show()

    prog_bar.update(1)

prog_bar.close()

fig, ax = plt.subplots(1, 2)
ax[0].imshow(target.detach().cpu().numpy()[0], cmap='gray')
ax[0].set_title('Target')
ax[1].imshow(approx.detach().cpu().numpy()[0], cmap='gray')
ax[1].set_title('Approximation')
plt.suptitle(f'Final approximation')
plt.show()

plt.plot(losses)
plt.title('Loss over iterations')
plt.show()

print(f'final latent vector: \n{latent_noise}')
