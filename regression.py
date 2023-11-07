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

from threading import Thread

torch.manual_seed(42)


def get_targets_and_labels(n_targets: int, target_class: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    :param n_targets: number of test images to return
    :param target_class: int between 0 and 9 to specify which digit to search. If None: all digits are considered
    :return: mnist test images and corresponding labels as np.ndarrays
    """
    # load test dataset
    project_root = os.getcwd()
    test_dataset, _ = get_rotated_mnist_dataloader(root=project_root,
                                                   batch_size=64,
                                                   shuffle=True,
                                                   one_hot_encode=True,
                                                   num_examples=10000,
                                                   num_rotations=0,
                                                   train=False)
    if target_class is not None:
        indices = np.where(test_dataset.targets == target_class)[0]
    else:
        indices = np.arange(n_targets)
    images = test_dataset.data[indices][:n_targets]
    labels = test_dataset.targets[indices][:n_targets]

    return images, labels


def plot_comparison(tar: torch.Tensor, approx: torch.Tensor, title: str):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(tar.detach().cpu().numpy(), cmap='gray')
    ax[0].set_title('Target')
    ax[1].imshow(approx.detach().cpu().numpy(), cmap='gray')
    ax[1].set_title('Approximation')
    plt.suptitle(title)
    plt.show()


def single_regression(generator: generators.Generator,
                      n_iterations: int,
                      tar: torch.Tensor,
                      trans,
                      class_to_search: int,
                      lr: float = 1e-2,
                      weight_decay: float = 1.0,
                      plot: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    # noise = torch.randn(1, LATENT_DIM, requires_grad=True, dtype=torch.float32, device=device)

    label = torch.zeros(1, 10, device=device)
    label[0, class_to_search] = 1

    tar = trans(tar).squeeze()
    tar = tar + torch.randn((IMG_SIZE, IMG_SIZE)) * 0.0
    tar = tar.to(device)
    # tar = torch.rot90(tar, 2, dims=(0, 1))

    criterion = torch.nn.MSELoss(reduction='sum')
    zero_noise = torch.zeros(1, LATENT_DIM, requires_grad=True, dtype=torch.float32, device=device)

    potential_start_vectors = [zero_noise]

    for i in range(15):
        noise_vec = 1.0 * torch.randn(1, LATENT_DIM, dtype=torch.float32)
        potential_start_vectors.append(noise_vec.to(device))

    start_vector_losses = []
    start_images = []

    for vec in potential_start_vectors:
        with torch.no_grad():
            approx = generator(vec, label).squeeze()
            loss = criterion(approx, tar)
            start_vector_losses.append(loss)
            start_images.append(approx)

    min_start_loss = start_vector_losses.index(min(start_vector_losses))
    noise = potential_start_vectors[min_start_loss].detach()
    noise.requires_grad = True

    optim = torch.optim.Adam(params=[noise], lr=lr, weight_decay=weight_decay)

    losses = []
    magnitudes = []

    if plot is True:
        fig, axs = plt.subplots(4, 5, figsize=(10, 8))

        # Plot the target image on the left
        axs[0, 0].imshow(tar.detach().cpu().numpy(), cmap='gray')
        axs[0, 0].set_title('Target')
        axs[0, 0].axis('off')

        for i in range(1, 4):
            axs[i, 0].axis('off')

        # Plot the 16 potential starting images on the right
        for i in range(4):
            for j in range(4):
                ax = axs[i, j + 1]
                ax.imshow(start_images[i * 4 + j], cmap='gray')
                ax.set_title(f'Start pos {i * 4 + j}')
                ax.axis('off')

        # highlight selected starting point
        highlighted_ax = axs[min_start_loss // 4, (min_start_loss % 4) + 1]
        highlighted_ax.axis('on')
        for orientation in highlighted_ax.spines:
            highlighted_ax.spines[orientation].set(color='red', linewidth=3)

        # Adjust spacing and display the plot
        plt.tight_layout()
        plt.show()

    # REGRESSION LOOP
    for i in range(n_iterations):
        optim.zero_grad()

        approx = generator(noise, label).squeeze()
        loss = criterion(approx, tar)

        loss.backward()
        optim.step()

        losses.append(loss.detach().cpu().numpy())
        mag = torch.linalg.norm(noise)
        magnitudes.append(mag.detach().cpu().numpy())

        if prog_bar:
            prog_bar.update(1)

        if plot and i % step_for_plot == 0:
            plot_comparison(tar, approx, f'iteration {i}')

    if plot is True:
        plt.plot(losses)
        plt.title('loss over iterations')
        plt.show()

        plt.plot(magnitudes)
        plt.title('magnitude of latent noise over iterations')
        plt.show()

    return noise, tar


def multiple_regressions(targets_numpy: np.ndarray, c_labels: np.ndarray, n_iterations, trans, start_idx, path, lr, wd):
    generator, _ = load_gen_disc_from_checkpoint(checkpoint_path=path, device=device, print_to_console=False)

    for target_idx in range(targets_numpy.shape[0]):
        tar = targets_numpy[target_idx]

        latent_noise, tar_tensor = single_regression(generator=generator,
                                                     n_iterations=n_iterations,
                                                     tar=tar,
                                                     trans=trans,
                                                     class_to_search=c_labels[target_idx],
                                                     lr=lr,
                                                     weight_decay=wd,
                                                     plot=False)

        LATENT_NOISE_RESULTS[start_idx + target_idx] = latent_noise.detach()
        ALL_TARGETS[start_idx + target_idx] = tar_tensor


def threaded_regression(n_threads, n_regressions, n_iterations, checkpoint_path, targets, c_labels, lr=1e-2,
                        weight_decay=1.0):
    assert n_regressions % n_threads == 0

    threads = []
    n_reg_per_thread = n_regressions // n_threads

    for i in range(n_threads):
        start = i * n_reg_per_thread
        end = (i + 1) * n_reg_per_thread
        t = Thread(target=multiple_regressions,
                   args=(
                       targets[start:end],
                       c_labels[start:end],
                       n_iterations,
                       transform,
                       start,
                       checkpoint_path,
                       lr,
                       weight_decay)
                   )
        threads.append(t)

    print(f'Starting {n_threads} threads \n')
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def plot_PSNRs(generator: generators.Generator, noise, c_labels, ground_truth):
    one_hot_labels = torch.nn.functional.one_hot(torch.from_numpy(c_labels), 10).type(torch.float32)

    all_predictions = generator(noise.to(device), one_hot_labels.to(device))
    snrs = peak_signal_noise_ratio(preds=all_predictions,
                                   target=ground_truth,
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

    plot_comparison(ground_truth[idx_worst_snr, 0], all_predictions[idx_worst_snr, 0],
                    title=f'Worst approximation. SNR: {snrs[idx_worst_snr].item():.2f}')
    plot_comparison(ground_truth[idx_best_snr, 0], all_predictions[idx_best_snr, 0],
                    title=f'Best approximation. SNR: {snrs[idx_best_snr].item():.2f}')

    print(f'index of worst approx: {idx_worst_snr}')
    print(f'index of best approx: {idx_best_snr}')


def save_regression_results(latent_noise, path_to_model, class_to_search, lr, weight_decay, img_size, latent_dim,
                            n_regressions, n_iterations, trans, description='-'):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_path = f'regressions/{current_date}'
    d = {
        'latent_noise': latent_noise,
        'path_to_model': path_to_model,
        'class_to_search': class_to_search,
        'lr': lr,
        'weight_decay': weight_decay,
        'img_size': img_size,
        'latent_dim': latent_dim,
        'n_regressions': n_regressions,
        'n_iterations': n_iterations,
        'transform': trans,
        'description': description
    }
    torch.save(d, save_path)
    print(f'saved regression results as {save_path}')


device = 'cpu'

# for conditional generators we need to specify which class we're looking at
CLASS_TO_SEARCH = None

# Hyperparameters
LR = 1e-2
WEIGHT_DECAY = 1.0
IMG_SIZE = 28
LATENT_DIM = 64

N_REGRESSIONS = 5
N_ITERATIONS = 10

step_for_plot = 30

CHECKPOINT_PATH = 'trained_models/p4_rot_mnist/2023-10-31_14:16:50/checkpoint_20000'
gen, _ = load_gen_disc_from_checkpoint(checkpoint_path=CHECKPOINT_PATH, device=device)
print_checkpoint(load_checkpoint(path=CHECKPOINT_PATH, device=device))

target_images, labels = get_targets_and_labels(N_REGRESSIONS, CLASS_TO_SEARCH)

LATENT_NOISE_RESULTS = torch.zeros(N_REGRESSIONS, LATENT_DIM, dtype=torch.float32, device=device)
ALL_TARGETS = torch.zeros(N_REGRESSIONS, 1, IMG_SIZE, IMG_SIZE, device=device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(IMG_SIZE),
     transforms.Normalize((0.5,), (0.5,))]
)

# setup progress bar
prog_bar = tqdm(total=N_ITERATIONS * N_REGRESSIONS)
prog_bar.set_description(f'Total of {N_REGRESSIONS} regressions with {N_ITERATIONS} iterations each')

# target = target_images[0] # 62
# single_regression(gen, 100, target, transform, CLASS_TO_SEARCH, LR, WEIGHT_DECAY, plot=True)

multiple_regressions(targets_numpy=target_images, c_labels=labels, n_iterations=N_ITERATIONS,
                     trans=transform, start_idx=0, path=CHECKPOINT_PATH, lr=LR, wd=WEIGHT_DECAY)

# threaded_regression(n_threads=10, n_regressions=N_REGRESSIONS, n_iterations=N_ITERATIONS, checkpoint_path=CHECKPOINT_PATH, targets=target_images, c_labels=labels, lr=LR, weight_decay=WEIGHT_DECAY)

plot_PSNRs(gen, LATENT_NOISE_RESULTS, labels, ALL_TARGETS)
save_regression_results(latent_noise=LATENT_NOISE_RESULTS,
                        path_to_model=CHECKPOINT_PATH,
                        class_to_search=CLASS_TO_SEARCH,
                        lr=LR,
                        weight_decay=WEIGHT_DECAY,
                        img_size=IMG_SIZE,
                        latent_dim=LATENT_DIM,
                        n_regressions=N_REGRESSIONS,
                        n_iterations=N_ITERATIONS,
                        trans=transform,
                        description='-')
prog_bar.close()
