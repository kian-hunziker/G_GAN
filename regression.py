import os

import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.functional.image import peak_signal_noise_ratio

import random
from tqdm import tqdm

from utils.dataLoaders import get_rotated_mnist_dataloader
from utils.checkpoints import load_checkpoint


def plot_comparison(target: torch.Tensor, approx: torch.Tensor, title: str):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(target.detach().cpu().numpy(), cmap='gray')
    ax[0].set_title('Target')
    ax[1].imshow(approx.detach().cpu().numpy(), cmap='gray')
    ax[1].set_title('Approximation')
    plt.suptitle(title)
    plt.show()

device = 'cpu'
# fix random seed for target selection. Torch seed is not fixed.
random.seed(2)

# for conditional generators we need to specify which class we're looking at
class_to_search = 0

# Hyperparameters
lr = 1e-2
weight_decay = 1.0
n_iterations = 100
img_size = 28
latent_dim = 100

n_regressions = 100

step_for_plot = n_iterations
plot_individual_loss_and_mag = False

gen, _ = load_checkpoint('trained_models/vanilla_small/2023-10-26 14:35:51/checkpoint_7000', device=device)
'''
gen = Generator()
gen.load_state_dict(
    torch.load('trained_models/z2_rot_mnist/2023-10-13 18:23:50/generator_test',
               map_location=device)
)
gen.eval()
'''

latent_noise_matrix = torch.zeros(n_regressions, latent_dim, dtype=torch.float)

project_root = os.path.dirname(os.path.abspath(__file__))
test_dataset, _ = get_rotated_mnist_dataloader(root=project_root,
                                               batch_size=64,
                                               shuffle=True,
                                               one_hot_encode=True,
                                               num_examples=10000,
                                               num_rotations=0,
                                               train=False)

indices_of_target_class = np.where(test_dataset.targets == class_to_search)[0]
targets = test_dataset.data[indices_of_target_class][:n_regressions]

all_targets = torch.zeros(n_regressions, 1, img_size, img_size)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(img_size),
     # transforms.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.8, 1), shear=[10,10,10,10]),
     transforms.Normalize((0.5,), (0.5,))]
)

prog_bar = tqdm(total=n_iterations * n_regressions)
prog_bar.set_description(f'Total of {n_regressions} regressions with {n_iterations} iterations each')

for target_idx in range(n_regressions):
    latent_noise = torch.zeros(1, latent_dim, requires_grad=True, dtype=torch.float, device=device)
    label = torch.zeros(1, 10, device=device)
    label[0, class_to_search] = 1
    target = targets[target_idx]
    target = transform(target).squeeze()
    target = target + torch.randn((img_size, img_size)) * 0.0
    target = target.to(device)

    optim = torch.optim.Adam(params=[latent_noise], lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss(reduction='sum')

    losses = []
    magnitudes = []

    for i in range(n_iterations):
        optim.zero_grad()

        approx = gen(latent_noise, label).squeeze()
        loss = criterion(approx, target)

        losses.append(loss.detach().cpu().numpy())
        mag = torch.linalg.vector_norm(latent_noise)
        magnitudes.append(mag.detach().cpu().numpy())

        loss.backward()
        optim.step()

        if i > 0 and i % step_for_plot == 0:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(target.detach().cpu().numpy(), cmap='gray')
            ax[0].set_title('Target')
            ax[1].imshow(approx.detach().cpu().numpy(), cmap='gray')
            ax[1].set_title('Approximation')
            plt.suptitle(f'Iteration {i} / {n_iterations}')
            plt.tight_layout()
            plt.show()

        prog_bar.update(1)

    latent_noise_matrix[target_idx] = latent_noise.detach()
    all_targets[target_idx, 0] = target

    if plot_individual_loss_and_mag is True:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(target.detach().cpu().numpy(), cmap='gray')
        ax[0].set_title('Target')
        ax[1].imshow(approx.detach().cpu().numpy(), cmap='gray')
        ax[1].set_title('Approximation')
        plt.suptitle(f'Final approximation')
        plt.show()

        plt.plot(losses)
        plt.title('Loss over iterations')
        plt.show()

        plt.plot(magnitudes)
        plt.title('Magnitude of latent noise over iterations')
        plt.show()

#print(f'final latent vector: \n{latent_noise}')
prog_bar.close()
all_labels = torch.zeros(n_regressions, 10)
all_labels[:, class_to_search] = torch.ones(n_regressions)


all_predictions = gen(latent_noise_matrix.to(device), all_labels.to(device))
snrs = peak_signal_noise_ratio(preds=all_predictions,
                               target=all_targets,
                               reduction='none',
                               dim=(1, 2, 3),
                               data_range=(-1, 1))

snrs_numpy = snrs.detach().cpu().numpy()

plt.plot(np.arange(n_regressions, dtype=int), snrs_numpy, 'o')
plt.xticks(np.arange(n_regressions, dtype=int))
plt.title('Peak Signal-To-Noise_Ratios')
plt.show()

sns.histplot(snrs_numpy)
plt.title(f'Histogram of PSNRs, total number of examples: {len(snrs_numpy)}')
plt.xlabel('PSNR values')
plt.show()

idx_worst_snr = torch.argmin(snrs).item()
idx_best_snr = torch.argmax(snrs).item()

plot_comparison(all_targets[idx_worst_snr, 0], all_predictions[idx_worst_snr, 0], title=f'Worst approximation. SNR: {snrs[idx_worst_snr].item():.2f}')
plot_comparison(all_targets[idx_best_snr, 0], all_predictions[idx_best_snr, 0], title=f'Best approximation. SNR: {snrs[idx_best_snr].item():.2f}')
