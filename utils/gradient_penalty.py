import torch

import discriminators


def zero_centered_gp_real_data(disc: discriminators.Discriminator,
                               real_batch: torch.Tensor,
                               labels: torch.Tensor,
                               device: torch.device | str):
    new_real_batch = 1.0 * real_batch
    new_real_batch.requires_grad = True
    new_labels = 1.0 * labels
    new_real_batch = new_real_batch.to(device)
    new_labels = new_labels.to(device)
    disc_opinion_real_new = disc([new_real_batch, new_labels])

    gradient = torch.autograd.grad(
        inputs=new_real_batch,
        outputs=disc_opinion_real_new,
        grad_outputs=torch.ones_like(disc_opinion_real_new),
        create_graph=True,
        retain_graph=True
    )[0]
    # gradient.shape: [batch_size, channels, height, width]
    gradient = gradient.view(gradient.shape[0], -1)
    # gradient.shape: [batch_size, channels * height * width]
    gradient_squared = torch.square(gradient)
    # gradient_norm = gradient.norm(2, dim=1)
    grad_square_sum = torch.sum(gradient_squared, dim=1)
    gradient_penalty = 0.5 * torch.mean(grad_square_sum)
    return gradient_penalty


def vanilla_gp(disc: discriminators.Discriminator,
               real_batch: torch.Tensor,
               fake_batch: torch.Tensor,
               device: torch.device | str):
    BATCH_SIZE, C, H, W = real_batch.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real_batch * epsilon + fake_batch * (1 - epsilon)
    interpolated_images.requires_grad = True

    # calculate critic scores
    mixed_scores = disc([interpolated_images, None])

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
