import datetime
import os
from argparse import ArgumentParser

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.functional.image import peak_signal_noise_ratio

from generators import Generator

from tqdm import tqdm


def get_trace_single_regression(gen: Generator, start: torch.Tensor, target: torch.Tensor,
                                label: int, n_iter: int = 100, lr: float = 0.01, wd: float = 1.0):
    assert isinstance(label, int)
    target = target.squeeze().unsqueeze(0)

    losses = []
    coords = []
    coords.append(1.0 * start.squeeze().detach().cpu())
    input_noise = 1.0 * start.squeeze()
    input_noise.requires_grad = True
    label = torch.nn.functional.one_hot(torch.tensor(label), 10).type(torch.float32).unsqueeze(0)

    optim = torch.optim.Adam([input_noise], lr=lr, weight_decay=wd)

    progbar = tqdm(total=n_iter)

    for i in range(n_iter):
        optim.zero_grad()

        approx = gen(input_noise.unsqueeze(0), label).squeeze().unsqueeze(0)
        loss = torch.nn.functional.mse_loss(approx, target, reduction='sum')
        with torch.no_grad():
            loss_trace = torch.nn.functional.mse_loss(approx, target)

        loss.backward()
        optim.step()

        losses.append(loss_trace.detach().cpu())
        coords.append(1.0 * input_noise.detach().cpu())

        progbar.update(1)

    final_loss = torch.nn.functional.mse_loss(approx, target)
    losses.append(final_loss.detach().cpu())
    coords = torch.stack(coords)

    x = coords.numpy()[:, 0]
    y = coords.numpy()[:, 1]
    z = torch.stack(losses).numpy()

    progbar.close()

    return [x, y, z]

