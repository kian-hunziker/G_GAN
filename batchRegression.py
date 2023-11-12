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
    ax[0].imshow(tar.squeeze().detach().cpu().numpy(), cmap='gray')
    ax[0].set_title('Target')
    ax[1].imshow(approx.squeeze().detach().cpu().numpy(), cmap='gray')
    ax[1].set_title('Approximation')
    plt.suptitle(title)
    plt.show()


def save_regression_results(path_to_model, latent_noise, snrs, loss_per_batch, class_to_search, lr, weight_decay,
                            img_size, latent_dim, n_regressions, n_iterations, n_start_pos, gen_arch, description='-'):
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
        'n_start_pos': n_start_pos,
        'description': description
    }
    torch.save(d, save_path)
    print(f'saved regression results as {save_path}')


arg_parser = ArgumentParser()
arg_parser.add_argument('-path', type=str)
arg_parser.add_argument('-batch_size', default=64, type=int)
arg_parser.add_argument('-n_batches', default=None, type=int)
arg_parser.add_argument('-n_iter', default=300, type=int)
arg_parser.add_argument('-n_start_pos', default=128, type=int)
arg_parser.add_argument('-plot', default=False, type=bool)
args = arg_parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fix random seed
torch.manual_seed(43)

LR = 0.1
WEIGHT_DECAY = 0.0
IMG_SIZE = 28

N_ITERATIONS = args.n_iter
BATCH_SIZE = args.batch_size
N_START_POS = args.n_start_pos
if args.n_batches is None:
    N_BATCHES = int(np.ceil(10000 / BATCH_SIZE))
else:
    N_BATCHES = args.n_batches
PLOT = args.plot
step_for_plot = 100

# initialise lists to save batch results
all_latent_noise_list = []
final_losses = []
snrs_list = []

# load generator from specified checkpoint
checkpoint_path = args.path
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
print(f'number of start positions: {N_START_POS}')
print(f'learning rate: {LR}')
print(f'weight decay: {WEIGHT_DECAY}')
print(f'latent dimension: {LATENT_DIM}')
print(f'device: {device}')
print('-' * 32 + '\n')

# setup progress bar and convert loader to iterable
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
    if N_START_POS > 1:
        # create potential start positions
        # first position is zero vector, the remaining N_START_POS - 1 are random inputs
        zero_start = torch.zeros(curr_batch_size, LATENT_DIM, dtype=torch.float32, device=device)
        random_starts = torch.randn((N_START_POS - 1) * curr_batch_size, LATENT_DIM, dtype=torch.float32, device=device)
        pot_start_vecs = torch.cat((zero_start, random_starts))

        # pass potential start vectors through generator and compute initial losses
        # chose start vector with minimal MSE for each example
        # do this computation for each example separately to avoid memory issues for high values of N_START_POS
        start_vecs = []
        with torch.no_grad():
            for i in range(curr_batch_size):
                start_idx = i * N_START_POS
                end_idx = (i + 1) * N_START_POS

                # compute approximations for current example
                start_approx = gen(pot_start_vecs[start_idx:end_idx], input_labels[i].unsqueeze(0).repeat_interleave(N_START_POS, dim=0))

                # compute loss
                start_losses = torch.nn.functional.mse_loss(
                    start_approx.squeeze(), target_images[i].repeat_interleave(N_START_POS, dim=0), reduction='none'
                )
                start_losses = torch.mean(start_losses, dim=(1, 2))

                # chose start vector with minimal loss
                idx = torch.argmin(start_losses).item()
                start_vecs.append(pot_start_vecs[start_idx + idx])

        input_noise = torch.stack(start_vecs)
    else:
        input_noise = torch.zeros(curr_batch_size, LATENT_DIM, device=device, dtype=torch.float32)
    input_noise.requires_grad = True

    # initialise loss function, optimizer and scheduler
    criterion = torch.nn.MSELoss(reduction='mean')
    optim = torch.optim.Adam([input_noise], lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.95)

    approx = None
    losses = []

    # main regression loop
    for i in range(N_ITERATIONS):
        optim.zero_grad()

        approx = gen(input_noise, input_labels)
        loss = criterion(approx, target_images)

        loss.backward()
        optim.step()
        scheduler.step()

        progbar.update(1)
        losses.append(loss.detach().cpu().numpy())

        if PLOT is True and (i % step_for_plot == 0):
            title = f'iteration : {i}\n current loss: {loss.detach().cpu().numpy():.4f}'
            plot_comparison(tar=target_images[0], approx=approx[0], title=title)

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

if PLOT is True:
    plt.plot(losses)
    plt.title(f'loss over iterations, lr: {LR}')
    plt.show()

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
                        n_start_pos=N_START_POS,
                        gen_arch=checkpoint['gen_arch'],
                        description='-')

print('\nDONE \n')
'''
_, new_loader = get_rotated_mnist_dataloader(root=project_root,
                                             batch_size=N_BATCHES * BATCH_SIZE,
                                             shuffle=False,
                                             one_hot_encode=True,
                                             num_examples=10000,
                                             num_rotations=0,
                                             train=False)


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