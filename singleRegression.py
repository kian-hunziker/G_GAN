import datetime
import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.functional.image import peak_signal_noise_ratio

from generators import Generator
from utils.checkpoints import load_gen_disc_from_checkpoint
from utils.dataLoaders import get_rotated_mnist_dataloader

from tqdm import tqdm


def plot_loss_and_comparison(loss_for_plot, target, approximation, title):
    if isinstance(loss_for_plot, list):
        loss_for_plot = np.array(loss_for_plot)

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 4, 3)
    ax3 = plt.subplot(1, 4, 4)
    axes = [ax1, ax2, ax3]
    sns.lineplot(ax=axes[0], data=np.log(loss_for_plot))
    axes[1].imshow(target.squeeze().detach().cpu().numpy(), cmap='gray')
    axes[2].imshow(approximation.squeeze().detach().cpu().numpy(), cmap='gray')

    axes[0].set_title('log(MSE) over iterations')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('log(MSE)')
    axes[1].set_title('Target')
    axes[2].set_title('Approximation')

    plt.suptitle(title, horizontalalignment='center', fontsize='xx-large')
    plt.show()


def single_regression(gen: Generator, start: torch.Tensor, target: torch.Tensor,
                      label: int, n_iter: int = 100, lr: float = 0.1, wd: float = 0.0,
                      step_for_plot: int = 0):
    assert isinstance(label, int)
    target = target.squeeze().unsqueeze(0)

    losses = []

    input_noise = 1.0 * start.squeeze()
    input_noise.requires_grad = True
    label = F.one_hot(torch.tensor(label), 10).type(torch.float32).unsqueeze(0)

    optim = torch.optim.Adam([input_noise], lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.95)

    progbar = tqdm(total=n_iter)
    progbar.set_description(f'Regression with {n_iter} iterations')

    for i in range(n_iter):
        optim.zero_grad()

        approx = gen(input_noise.unsqueeze(0), label).squeeze().unsqueeze(0)
        loss = F.mse_loss(approx, target, reduction='mean')

        loss.backward()
        optim.step()
        scheduler.step()

        losses.append(loss.detach().cpu().numpy())

        progbar.update(1)

        if step_for_plot > 0 and i % step_for_plot == 0:
            title = f'Iteration: {i}/{n_iter}'
            plot_loss_and_comparison(loss_for_plot=losses, target=target, approximation=approx, title=title)

    final_loss = F.mse_loss(approx, target)
    losses.append(final_loss.detach().cpu().numpy())

    title = f'Regression results after {n_iter} iterations'
    plot_loss_and_comparison(loss_for_plot=losses, target=target, approximation=approx, title=title)

    progbar.close()

    return input_noise.detach(), losses


def get_trace_single_regression(gen: Generator, start: torch.Tensor, target: torch.Tensor,
                                label: int, n_iter: int = 100, lr: float = 0.01, wd: float = 0.0):
    assert isinstance(label, int)
    assert start.shape[0] == 2
    target = target.squeeze().unsqueeze(0)

    losses = []
    coords = [1.0 * start.squeeze().detach().cpu()]
    input_noise = 1.0 * start.squeeze()
    input_noise.requires_grad = True
    label = F.one_hot(torch.tensor(label), 10).type(torch.float32).unsqueeze(0)

    optim = torch.optim.Adam([input_noise], lr=lr, weight_decay=wd)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.95)

    progbar = tqdm(total=n_iter)
    progbar.set_description(f'Regression with {n_iter}iterations')

    for i in range(n_iter):
        optim.zero_grad()

        approx = gen(input_noise.unsqueeze(0), label).squeeze().unsqueeze(0)
        loss = F.mse_loss(approx, target, reduction='mean')

        loss.backward()
        optim.step()

        losses.append(loss.detach().cpu())
        coords.append(1.0 * input_noise.detach().cpu())

        progbar.update(1)

    final_loss = F.mse_loss(approx, target)
    losses.append(final_loss.detach().cpu())
    coords = torch.stack(coords)

    x = coords.numpy()[:, 0]
    y = coords.numpy()[:, 1]
    z = torch.stack(losses).numpy()

    progbar.close()

    return [x, y, z]


def get_start_position(generator: Generator, latent_dim: int, target: torch.Tensor, label: int, n_start_pos: int = 64):
    assert isinstance(label, int)
    labels = F.one_hot(torch.tensor(label), 10).type(torch.float32).unsqueeze(0).repeat_interleave(n_start_pos, dim=0)
    targets = target.squeeze().unsqueeze(0).unsqueeze(0).repeat_interleave(n_start_pos, dim=0)
    zero_start = torch.zeros(1, latent_dim, dtype=torch.float32)
    random_positions = torch.randn(n_start_pos - 1, latent_dim, dtype=torch.float32)
    pot_start_positions = torch.cat((zero_start, random_positions), dim=0)

    with torch.no_grad():
        approx = generator(pot_start_positions, labels)
        loss = torch.mean(F.mse_loss(approx, targets, reduction='none'), dim=(1, 2, 3))
    min_idx = torch.argmin(loss).item()
    start_vec = pot_start_positions[min_idx]

    x = pot_start_positions[:, 0].detach().numpy()
    y = pot_start_positions[:, 1].detach().numpy()
    z = loss.detach().numpy()

    return start_vec, [x, y, z]


def foo():
    project_root = os.getcwd()
    test_dataset, loader = get_rotated_mnist_dataloader(root=project_root,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        one_hot_encode=True,
                                                        num_examples=10000,
                                                        num_rotations=0,
                                                        train=False,
                                                        single_class=None)

    loader_iterator = iter(loader)
    target_images, one_hot_label = next(loader_iterator)
    label = int(torch.where(one_hot_label.squeeze() == 1)[0])
    print(f'label: {label}')
    gen, _ = load_gen_disc_from_checkpoint('trained_models/z2_rot_mnist/2023-10-31_12:34:55/checkpoint_20000')

    start_pos, _ = get_start_position(gen, 64, target_images, label)
    lat_noise, loss = single_regression(gen, start_pos, target_images, label, n_iter=1000, step_for_plot=100)

