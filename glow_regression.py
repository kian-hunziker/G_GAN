import os

import torch
import torch.nn.functional as F
import numpy as np
import normflows as nf
import matplotlib.pyplot as plt
import seaborn as sns

from utils.checkpoints import load_glow_from_checkpoint
from utils.data_loaders import get_rotated_mnist_dataloader

from tqdm import tqdm


def glow_regression(gen: nf.core.MultiscaleFlow, target, label, n_iter, lr, wd, zero_start=False,
                    scheduler_step_size=None):
    images = []
    losses = []

    target.squeeze().unsqueeze(0).unsqueeze(0)

    if scheduler_step_size is None:
        scheduler_step_size = n_iter

    if zero_start is True:
        z = [torch.zeros(1, 16, 2, 2, requires_grad=True),
             torch.zeros(1, 4, 4, 4, requires_grad=True),
             torch.zeros(1, 2, 8, 8, requires_grad=True)]
    else:
        z = []
        num_samples = 10000
        rep_label = label.repeat(num_samples)
        for q in gen.q0:
            samples = q(num_samples, rep_label)[0]
            mean = torch.mean(samples, dim=0).unsqueeze(0).detach()
            mean.requires_grad = True
            z.append(mean)

    optim = torch.optim.Adam(params=z, lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=scheduler_step_size, gamma=0.95)

    for i in tqdm(range(n_iter)):
        optim.zero_grad()

        approx, _ = gen.forward_and_log_det(z)
        loss = F.mse_loss(approx, target, reduction='mean')

        loss.backward()
        optim.step()
        scheduler.step()

        losses.append(loss.detach().cpu().numpy())
        images.append(approx.squeeze().detach().cpu().numpy())

    return z, images, losses


def glow_reg_test():
    project_root = os.getcwd()
    test_dataset, loader = get_rotated_mnist_dataloader(root=project_root,
                                                        batch_size=10000,
                                                        shuffle=False,
                                                        one_hot_encode=False,
                                                        num_examples=10000,
                                                        num_rotations=0,
                                                        train=False,
                                                        img_size=16,
                                                        single_class=None,
                                                        glow=True)

    test_iter = iter(loader)
    target_images, labels = next(test_iter)

    model = load_glow_from_checkpoint('trained_models/glow/2023-11-23_12:25:04/checkpoint_30000')

    z_final, im, l = glow_regression(model, target_images[0].unsqueeze(0), labels[0], 10, lr=1e-2, wd=0)
    print(f'z shape: {len(z_final)}')
    for coord in z_final:
        print(coord.shape)
    print(f'n images: {len(im)}')
    print(f'image shape: {im[0].shape}')
    print(f'loss_shape: {len(l)}')

    plt.plot(l)
    plt.show()


#glow_reg_test()
