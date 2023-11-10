import datetime
import os
from argparse import ArgumentParser

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.functional.image import peak_signal_noise_ratio

from tqdm import tqdm

from utils.dataLoaders import get_rotated_mnist_dataloader
from utils.checkpoints import load_gen_disc_from_checkpoint, load_checkpoint, print_checkpoint


def plot_comparison(tar: torch.Tensor, approx: torch.Tensor, title: str):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(tar.detach().cpu().numpy(), cmap='gray')
    ax[0].set_title('Target')
    ax[1].imshow(approx.detach().cpu().numpy(), cmap='gray')
    ax[1].set_title('Approximation')
    plt.suptitle(title)
    plt.show()


def save_regression_results(path_to_model, latent_noise, snrs, loss_per_batch, class_to_search, lr, weight_decay,
                            img_size, latent_dim, n_regressions, n_iterations, gen_arch, description='-'):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_path = f'regressions/{current_date}_{gen_arch}'
    d = {
        'path_to_model': path_to_model,
        'latent_noise': latent_noise,
        'snrs': snrs,
        'loss_per_batch': loss_per_batch,
        'class_to_search': class_to_search,
        'lr': lr,
        'weight_decay': weight_decay,
        'img_size': img_size,
        'latent_dim': latent_dim,
        'n_regressions': n_regressions,
        'n_iterations': n_iterations,
        'description': description
    }
    torch.save(d, save_path)
    print(f'saved regression results as {save_path}')


arg_parser = ArgumentParser()
arg_parser.add_argument('-path', type=str)
arg_parser.add_argument('-batch_size', default=64, type=int)
arg_parser.add_argument('-n_batches', default=None, type=int)
arg_parser.add_argument('-n_iter', default=300, type=int)
args = arg_parser.parse_args()

device = 'cpu'
LR = 1e-2
WEIGHT_DECAY = 1.0
IMG_SIZE = 28

N_ITERATIONS = args.n_iter
BATCH_SIZE = args.batch_size
if args.n_batches is None:
    N_BATCHES = int(np.ceil(10000 / BATCH_SIZE))
else:
    N_BATCHES = args.n_batches

all_latent_noise_list = []
final_losses = []
snrs_list = []

checkpoint_path = args.path #'trained_models/p4_rot_mnist/2023-10-31_14:16:50/checkpoint_20000'
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

print('\n' + '-' * 32)
print(f'STARTING REGRESSION')
print(f'number of batches: {N_BATCHES}')
print(f'number of iterations per regression: {N_ITERATIONS}')
print(f'learning rate: {LR}')
print(f'weight decay: {WEIGHT_DECAY}')
print(f'latent dimension: {LATENT_DIM}')
print(f'device: {device}')
print('-' * 32 + '\n')


progbar = tqdm(total=N_BATCHES * N_ITERATIONS)
loader_iter = iter(loader)

# loop through batches
for batch_idx in range(N_BATCHES):
    # load target images and labels for current batch
    target_images, input_labels = next(loader_iter)
    target_images = target_images.to(device)
    input_labels = input_labels.to(device)

    curr_batch_size = target_images.shape[0]

    # initialise latent noise
    input_noise = torch.zeros(curr_batch_size, LATENT_DIM, dtype=torch.float32, requires_grad=True, device=device)

    # initialise loss function and optimizer
    criterion = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam([input_noise], lr=LR, weight_decay=WEIGHT_DECAY)

    approx = None

    # main regression loop
    for i in range(N_ITERATIONS):
        optim.zero_grad()

        approx = gen(input_noise, input_labels)
        loss = criterion(approx.squeeze(), target_images[:curr_batch_size].squeeze())

        loss.backward()
        optim.step()

        progbar.update(1)

    # compute SNRs and append results to lists
    final_losses.append(loss.detach().cpu().numpy())
    all_latent_noise_list.append(input_noise.detach().cpu())
    tmp_snrs = peak_signal_noise_ratio(preds=approx,
                                       target=target_images,
                                       reduction='none',
                                       dim=(1, 2, 3),
                                       data_range=(-1, 1))
    snrs_list.append(tmp_snrs.detach().cpu())

progbar.close()

# convert lists to torch.tensors
latent_noise_results = torch.cat(all_latent_noise_list)
all_snrs = torch.cat(snrs_list)

# save regression results
save_regression_results(path_to_model=checkpoint_path,
                        latent_noise=latent_noise_results,
                        snrs=all_snrs,
                        loss_per_batch=final_losses,
                        class_to_search=None,
                        lr=LR,
                        weight_decay=WEIGHT_DECAY,
                        img_size=IMG_SIZE,
                        latent_dim=LATENT_DIM,
                        n_regressions=latent_noise_results.shape[0],
                        n_iterations=N_ITERATIONS,
                        gen_arch=checkpoint['gen_arch'],
                        description='-')


_, new_loader = get_rotated_mnist_dataloader(root=project_root,
                                             batch_size=N_BATCHES * BATCH_SIZE,
                                             shuffle=False,
                                             one_hot_encode=True,
                                             num_examples=10000,
                                             num_rotations=0,
                                             train=False)

print('\nDONE \n')
'''
all_targets, all_labels = next(iter(new_loader))

sns.histplot(all_snrs.numpy())
plt.title(f'Histogram of PSNRs, total number of examples: {len(all_snrs.numpy())}')
plt.xlabel('PSNR values')
plt.show()

idx_worst_snr = torch.argmin(all_snrs).item()
idx_best_snr = torch.argmax(all_snrs).item()

worst_approx = gen(latent_noise_results[idx_worst_snr].unsqueeze(0), all_labels[idx_worst_snr].unsqueeze(0))
best_approx = gen(latent_noise_results[idx_best_snr].unsqueeze(0), all_labels[idx_best_snr].unsqueeze(0))

plot_comparison(all_targets[idx_worst_snr, 0], worst_approx[0, 0],
                title=f'Worst approximation. SNR: {all_snrs[idx_worst_snr].item():.2f}')
plot_comparison(all_targets[idx_best_snr, 0], best_approx[0, 0],
                title=f'Best approximation. SNR: {all_snrs[idx_best_snr].item():.2f}')

print(f'all snrs: {all_snrs}')

plt.plot(final_losses)
plt.title('loss over batches')
plt.show()
'''
