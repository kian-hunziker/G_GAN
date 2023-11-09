import datetime
import os

import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.functional.image import peak_signal_noise_ratio

from tqdm import tqdm

import generators
from utils.dataLoaders import get_rotated_mnist_dataloader
from utils.checkpoints import load_gen_disc_from_checkpoint, load_checkpoint, print_checkpoint

device = 'cpu'
LR = 1e-2
WEIGHT_DECAY = 1.0
IMG_SIZE = 28

N_ITERATIONS = 300
BATCH_SIZE = 2

checkpoint_path = 'trained_models/p4_rot_mnist/2023-11-09_18:14:22/checkpoint_20000'
gen, _ = load_gen_disc_from_checkpoint(checkpoint_path, device=device, print_to_console=True)
checkpoint = load_checkpoint(checkpoint_path)
print_checkpoint(checkpoint)
LATENT_DIM = checkpoint['latent_dim']
gen.eval()


# load test dataset
project_root = os.getcwd()
test_dataset, loader = get_rotated_mnist_dataloader(root=project_root,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False,
                                                    one_hot_encode=True,
                                                    num_examples=10000,
                                                    num_rotations=0,
                                                    train=False)

target_images, input_labels = next(iter(loader))
target_images = target_images.to(device)
input_labels = input_labels.to(device)

input_noise = torch.zeros(BATCH_SIZE, LATENT_DIM, dtype=torch.float32, requires_grad=True, device=device)

criterion = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam([input_noise], lr=LR, weight_decay=WEIGHT_DECAY)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.999)

losses = []
progbar = tqdm(total=N_ITERATIONS)

for i in range(N_ITERATIONS):
    optim.zero_grad()

    approx = gen(input_noise, input_labels)
    loss = criterion(approx.squeeze(), target_images[:BATCH_SIZE].squeeze())

    loss.backward()
    optim.step()

    losses.append(loss.detach().cpu().numpy())
    #scheduler.step()
    progbar.update(1)

progbar.close()

plt.plot(losses)
plt.title('loss over iterations')
plt.show()


def plot_comparison(tar: torch.Tensor, approx: torch.Tensor, title: str):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(tar.detach().cpu().numpy(), cmap='gray')
    ax[0].set_title('Target')
    ax[1].imshow(approx.detach().cpu().numpy(), cmap='gray')
    ax[1].set_title('Approximation')
    plt.suptitle(title)
    plt.show()


approx = gen(input_noise, input_labels)
for i in range(BATCH_SIZE):
    plot_comparison(target_images[i, 0], approx[i, 0], title=f'{i}')

print(input_noise)
