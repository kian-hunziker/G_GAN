import torch
import torch.nn as nn
import torch.optim as optim

import discriminators
import generators


def get_optimizers(lr_g, beta1_g, beta2_g, lr_d, beta1_d, beta2_d, gen: generators.Generator, disc: discriminators.Discriminator):
    """
    Function to return Adam optimizer objects. Note the calls to
    optimizer.iterations and optimizer.decay. They need to be called for TF2
    checkpointing for some reason.

    Args
        lr_g: float
            generator learning rate.
        beta1_g: float
            generator beta_1 Adam parameter.
        beta2_g: float
            generator beta_2 Adam parameter.
        lr_d: float
            discriminator learning rate.
        beta1_d: float
            discriminator beta_1 Adam parameter.
        beta2_d: float
            discriminator beta_2 Adam parameter.
    """
    generator_optimizer = optim.Adam(
        params=gen.parameters(),
        lr=lr_g,
        betas=(beta1_g, beta2_g),
        eps=1e-7,
    )
    # generator_optimizer.iterations
    # generator_optimizer.decay = tf.Variable(0.0)

    discriminator_optimizer = optim.Adam(
        params=disc.parameters(),
        lr=lr_d,
        betas=(beta1_d, beta2_d),
        eps=1e-7,
    )
    # discriminator_optimizer.iterations
    # discriminator_optimizer.decay = tf.Variable(0.0)

    return generator_optimizer, discriminator_optimizer
