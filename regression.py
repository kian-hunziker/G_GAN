import os

import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.functional.image import peak_signal_noise_ratio

import random
from tqdm import tqdm

import generators
from utils.dataLoaders import get_rotated_mnist_dataloader
from utils.checkpoints import load_gen_disc_from_checkpoint, load_checkpoint, print_checkpoint

from threading import Thread

device = 'cpu'
# fix random seed for target selection. Torch seed is not fixed.
random.seed(2)

# for conditional generators we need to specify which class we're looking at
CLASS_TO_SEARCH = 2

# Hyperparameters
LR = 1e-2
WEIGHT_DECAY = 1.0
N_ITERATIONS = 100
img_size = 28
latent_dim = 64

N_REGRESSIONS = 100

step_for_plot = N_ITERATIONS
plot_individual_loss_and_mag = False


def plot_comparison(target: torch.Tensor, approx: torch.Tensor, title: str):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(target.detach().cpu().numpy(), cmap='gray')
    ax[0].set_title('Target')
    ax[1].imshow(approx.detach().cpu().numpy(), cmap='gray')
    ax[1].set_title('Approximation')
    plt.suptitle(title)
    plt.show()


def single_regression(gen: generators.Generator, n_iterations: int, tar: torch.Tensor, class_to_search: int,
                      lr: float = 1e-2, weight_decay: float = 1.0) -> torch.Tensor:
    noise = torch.zeros(1, latent_dim, requires_grad=True, dtype=torch.float, device=device)
    label = torch.zeros(1, 10, device=device)
    label[0, class_to_search] = 1

    optim = torch.optim.Adam(params=[noise], lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss(reduction='sum')

    for i in range(n_iterations):
        optim.zero_grad()

        approx = gen(noise, label).squeeze()
        loss = criterion(approx, tar)

        loss.backward()
        optim.step()

        if prog_bar:
            prog_bar.update(1)

    return noise


def multiple_regression(targets_numpy: np.ndarray, transform, start_idx, path):
    generator, _ = load_gen_disc_from_checkpoint(checkpoint_path=path, device=device, print_to_console=False)

    for target_idx in range(targets_numpy.shape[0]):
        target = targets_numpy[target_idx]
        target = transform(target).squeeze()
        target = target + torch.randn((img_size, img_size)) * 0.0
        target = target.to(device)

        latent_noise = single_regression(gen=generator,
                                         n_iterations=N_ITERATIONS,
                                         tar=target,
                                         class_to_search=CLASS_TO_SEARCH,
                                         lr=LR,
                                         weight_decay=WEIGHT_DECAY)

        #prog_bar.update(N_ITERATIONS)

        latent_noise_matrix[start_idx + target_idx] = latent_noise.detach()
        all_targets[start_idx + target_idx] = target


checkpoint_path = 'trained_models/p4_rot_mnist/2023-10-31 14:16:50/checkpoint_20000'
gen, _ = load_gen_disc_from_checkpoint(checkpoint_path=checkpoint_path, device=device)
print_checkpoint(load_checkpoint(path=checkpoint_path, device=device))

# load test dataset
project_root = os.getcwd()
test_dataset, _ = get_rotated_mnist_dataloader(root=project_root,
                                               batch_size=64,
                                               shuffle=True,
                                               one_hot_encode=True,
                                               num_examples=10000,
                                               num_rotations=0,
                                               train=False)

indices_of_target_class = np.where(test_dataset.targets == CLASS_TO_SEARCH)[0]
targets = test_dataset.data[indices_of_target_class][:N_REGRESSIONS]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(img_size),
     # transforms.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.8, 1), shear=[10,10,10,10]),
     transforms.Normalize((0.5,), (0.5,))]
)



# setup progress bar
prog_bar = tqdm(total=N_ITERATIONS * N_REGRESSIONS)
prog_bar.set_description(f'Total of {N_REGRESSIONS} regressions with {N_ITERATIONS} iterations each')

latent_noise_matrix = torch.zeros(N_REGRESSIONS, latent_dim, dtype=torch.float32, device=device)
all_targets = torch.zeros(N_REGRESSIONS, 1, img_size, img_size, device=device)

n_threads = 4
assert N_REGRESSIONS % n_threads == 0
threads = []
n_reg_per_thread = N_REGRESSIONS // n_threads

for i in range(n_threads):
    start = i * n_reg_per_thread
    end = (i + 1) * n_reg_per_thread
    t = Thread(target=multiple_regression, args=(targets[start:end], transform, start, checkpoint_path))
    threads.append(t)

for t in threads:
    t.start()
for t in threads:
    t.join()

prog_bar.close()

# prepare labels for PSNR calculation
all_labels = torch.zeros(N_REGRESSIONS, 10)
all_labels[:, CLASS_TO_SEARCH] = torch.ones(N_REGRESSIONS)

all_predictions = gen(latent_noise_matrix.to(device), all_labels.to(device))
snrs = peak_signal_noise_ratio(preds=all_predictions,
                               target=all_targets,
                               reduction='none',
                               dim=(1, 2, 3),
                               data_range=(-1, 1))

snrs_numpy = snrs.detach().cpu().numpy()

sns.histplot(snrs_numpy)
plt.title(f'Histogram of PSNRs, total number of examples: {len(snrs_numpy)}')
plt.xlabel('PSNR values')
plt.show()

idx_worst_snr = torch.argmin(snrs).item()
idx_best_snr = torch.argmax(snrs).item()

plot_comparison(all_targets[idx_worst_snr, 0], all_predictions[idx_worst_snr, 0],
                title=f'Worst approximation. SNR: {snrs[idx_worst_snr].item():.2f}')
plot_comparison(all_targets[idx_best_snr, 0], all_predictions[idx_best_snr, 0],
                title=f'Best approximation. SNR: {snrs[idx_best_snr].item():.2f}')
