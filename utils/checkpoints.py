import torch
import normflows as nf
import numpy as np

import glow_models
from discriminators import Discriminator
from generators import Generator


def load_gen_disc_from_checkpoint(checkpoint_path, device='cpu', print_to_console=True) -> tuple[Generator, Discriminator]:
    """
    :param checkpoint_path: complete path of saved checkpoint
    :param device: device to load the networks to
    :param print_to_console: If True, the information fields of the checkpoint will be printed to the console
    :return: Generator, Discriminator, in eval() mode, loaded to device
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen_arch = checkpoint['gen_arch']
    disc_arch = checkpoint['disc_arch']
    latent_dim = checkpoint['latent_dim']
    gen = Generator(gen_arch=gen_arch, latent_dim=latent_dim)
    disc = Discriminator([1, 28, 28], disc_arch=disc_arch)
    gen.load_state_dict(checkpoint['generator'])
    disc.load_state_dict(checkpoint['discriminator'])
    gen.to(device)
    disc.to(device)
    gen.eval()
    disc.eval()

    if print_to_console is True:
        print('-' * 32)
        print(f'Loaded checkpoint from: {checkpoint_path}')
        print(f'Generator architecture: {gen_arch}')
        print(f'Discriminator architecture: {disc_arch}')
        print('-' * 32)
        print('\n')

    return gen, disc


def load_checkpoint(path: str, device: str | torch.device = 'cpu') -> dict:
    return torch.load(path, map_location=device)


def print_checkpoint(checkpoint: dict) -> None:
    for key, value in checkpoint.items():
        if not isinstance(value, dict):
            key = key + ': ' + '.' * (28 - len(key) - 2)
            print(f'{key : <28} {value}')
    print('\n')


def load_glow_from_checkpoint(path: str, device: str | torch.device = 'cpu', arch: str = 'cc'):
    if arch == 'cc':
        model = glow_models.get_cc_mnist_glow_model()
    elif arch == 'unconditional_mnist':
        model = glow_models.get_unconditional_mnist_glow_model()

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['generator'])
    model.to(device)
    model.eval()

    for key, value in checkpoint.items():
        if not isinstance(value, dict) and not isinstance(value, np.ndarray):
            key = key + ': ' + '.' * (28 - len(key) - 2)
            print(f'{key : <28} {value}')
    print('\n')

    return model
