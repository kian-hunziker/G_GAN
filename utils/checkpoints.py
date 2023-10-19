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
    gen = Generator(gen_arch=checkpoint['gen_arch'])
    disc = Discriminator([1, 28, 28], disc_arch=checkpoint['disc_arch'])
    gen.load_state_dict(checkpoint['generator'])
    disc.load_state_dict(checkpoint['discriminator'])
    gen.to(device)
    disc.to(device)
    gen.eval()
    disc.eval()
    return gen, disc