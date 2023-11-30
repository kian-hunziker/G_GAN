import numpy as np
import normflows as nf


def get_cc_mnist_glow_model():
    # Define flows
    L = 3
    K = 16

    image_size = 16
    input_shape = (1, image_size, image_size)
    n_dims = np.prod(input_shape)
    channels = 1
    hidden_channels = 256
    split_mode = 'channel'
    scale = True
    num_classes = 10

    # Set up flows, distributions and merge operations
    q0 = []
    merges = []
    flows = []
    for i in range(L):
        flows_ = []
        for j in range(K):
            flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                          split_mode=split_mode, scale=scale)]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i),
                            input_shape[2] // 2 ** (L - i))
        else:
            latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L,
                            input_shape[2] // 2 ** L)
        q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]

    # Construct flow model with the multiscale architecture
    model = nf.MultiscaleFlow(q0, flows, merges)
    return model


def get_unconditional_mnist_glow_model():
    # Define flows
    L = 3
    K = 16

    image_size = 16
    input_shape = (1, image_size, image_size)
    n_dims = np.prod(input_shape)
    channels = 1
    hidden_channels = 256
    split_mode = 'channel'
    scale = True

    # Set up flows, distributions and merge operations
    q0 = []
    merges = []
    flows = []
    for i in range(L):
        flows_ = []
        for j in range(K):
            flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                          split_mode=split_mode, scale=scale)]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i),
                            input_shape[2] // 2 ** (L - i))
        else:
            latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L,
                            input_shape[2] // 2 ** L)
        q0 += [nf.distributions.DiagGaussian(latent_shape)]

    # Construct flow model with the multiscale architecture
    model = nf.MultiscaleFlow(q0, flows, merges)
    return model
