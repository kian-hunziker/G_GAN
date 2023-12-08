from math import prod

import torch


def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def reshape_z_for_glow(z_vec, glow_instance):
    """
    Reshape latent vector z from vector representation to input format for GLOW model
    :param z_vec: latent vector [batch_size, latent_dim]
    :param glow_instance: GLOW model that has base distributions in glow_instance.q0
    :return: reshaped z vector that can be passed to glow_instance.forward_and_log_det(z)
    """
    z = []
    start = 0
    curr_batch_size = z_vec.shape[0]
    for q in glow_instance.q0:
        length = int(prod(q.shape))
        z_temp = z_vec[:, start:start + length]
        start += length
        z_temp = z_temp.reshape((curr_batch_size,) + q.shape)
        z.append(z_temp)
    return z
