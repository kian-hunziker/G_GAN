import torch

from discriminators import Discriminator
from generators import Generator


def load_checkpoint(checkpoint_path, device='cpu'):
    """
    :param checkpoint_path: complete path of saved checkpoint
    :param device: device to load the networks to
    :return: Generator, Discriminator, in eval() mode, loaded to device
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen_arch = checkpoint['gen_arch']
    disc_arch = checkpoint['disc_arch']
    if gen_arch == 'vanilla' or gen_arch == 'vanilla_small':
        latent_dim = 100
    else:
        latent_dim  = 64
    gen = Generator(gen_arch=gen_arch, latent_dim=latent_dim)
    disc = Discriminator([1, 28, 28], disc_arch=disc_arch)
    gen.load_state_dict(checkpoint['generator'])
    disc.load_state_dict(checkpoint['discriminator'])
    gen.to(device)
    disc.to(device)
    gen.eval()
    disc.eval()

    print('-' * 32)
    print(f'Loaded checkpoint from: {checkpoint_path}')
    print(f'Generator architecture: {gen_arch}')
    print(f'Discriminator architecture : {disc_arch}\n')
    print('-' * 32)
    print('\n')

    return gen, disc