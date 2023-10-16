import numpy as np
import torch
import torch.nn as nn


class GFiLM(nn.Module):
    def __init__(self, feature_map_shape: list, gamma_linear: nn.Linear, beta_linear: nn.Linear, group: str):
        """
        :param feature_map_shape: list [n_channels, height, width]
        :param gamma_linear: linear layer to compute gamma [proj_dim -> n_channels]
        :param beta_linear: linear layer to compute beta [proj_dim -> n_channels]
        :param group: group: one of {'Z2', 'C4'}
        """
        super(GFiLM, self).__init__()
        assert isinstance(feature_map_shape, list)
        FiLM_gamma_shape = gamma_linear.weight.shape
        FiLM_beta_shape = beta_linear.weight.shape

        self.gamma_linear = gamma_linear
        self.beta_linear = beta_linear

        self.height = feature_map_shape[1]
        self.width = feature_map_shape[2]
        self.n_channels = feature_map_shape[0]

        self.group = group

        assert (int(self.n_channels) == FiLM_gamma_shape[0])
        assert (int(self.n_channels) == FiLM_beta_shape[0])

    def forward(self, x: list) -> torch.Tensor:
        """
        :param x: [conv_output, cla]. conv_output is the output of the convolutional layer above.
                    for Z2: [batch_size, n_channels, height, width]
                    for c4: [batch_size, n_channels, group_dim, height, width]
                    cla: is the concatenated projected noise and label embedding. length = proj_dim
        :return: affine transformation of conv_input, same shape as conv_input
        """
        assert isinstance(x, list)
        conv_output, cla = x

        FiLM_gamma = self.gamma_linear(cla)
        FiLM_beta = self.beta_linear(cla)
        # FiLM_gamma: [batch_size, channels]
        # FiLM_beta: [batch_size, channels]
        if self.group == 'Z2':
            '''
            conv_output: [batch_size, channels, height, width]
            '''
            FiLM_gamma = torch.unsqueeze(FiLM_gamma, -1)
            FiLM_gamma = torch.unsqueeze(FiLM_gamma, -1)
            FiLM_gamma = torch.tile(FiLM_gamma, (self.height, self.width))

            FiLM_beta = torch.unsqueeze(FiLM_beta, -1)
            FiLM_beta = torch.unsqueeze(FiLM_beta, -1)
            FiLM_beta = torch.tile(FiLM_beta, (self.height, self.width))

            # apply affine transformation
            return (1 + FiLM_gamma) * conv_output + FiLM_beta

        elif self.group == 'C4':
            # conf output should have shape [batch_size, n_channels, group_dim, height, width]
            # C4 is all 90 degrees rotations, so the group_dim is 4
            group_dim = 4
            assert group_dim == conv_output.shape[2]
            # go from [batch_size, n_channels] to [batch_size, n_channels, 1, 1, 1]
            FiLM_gamma = torch.unsqueeze(FiLM_gamma, -1)
            FiLM_gamma = torch.unsqueeze(FiLM_gamma, -1)
            FiLM_gamma = torch.unsqueeze(FiLM_gamma, -1)
            # repeat entries and go to shape [batch_size, n_channels, group_dim, height, width]
            FiLM_gamma = torch.tile(FiLM_gamma, (group_dim, self.height, self.width))

            FiLM_beta = torch.unsqueeze(FiLM_beta, -1)
            FiLM_beta = torch.unsqueeze(FiLM_beta, -1)
            FiLM_beta = torch.unsqueeze(FiLM_beta, -1)
            FiLM_beta = torch.tile(FiLM_beta, (group_dim, self.height, self.width))

            # apply affine transformation
            return (1 + FiLM_gamma) * conv_output + FiLM_beta


def c4_test():
    batch_size = 5
    n_channels = 1
    img_dim = 28

    group_dim = 4
    FiLM_gamma = torch.Tensor(np.arange(batch_size)).unsqueeze(-1)
    FiLM_beta = torch.Tensor(np.arange(batch_size)).unsqueeze(-1)
    print(FiLM_gamma)
    print(f'gamma.shape: {FiLM_gamma.shape}, [batch_size, n_channels')
    conv_output = torch.ones((batch_size, n_channels, group_dim, img_dim, img_dim))
    print(f'conv output shape: {conv_output.shape}')

    assert group_dim == conv_output.shape[2]
    # go from [batch_size, n_channels] to [batch_size, n_channels, 1, 1, 1]
    FiLM_gamma = torch.unsqueeze(FiLM_gamma, -1)
    FiLM_gamma = torch.unsqueeze(FiLM_gamma, -1)
    FiLM_gamma = torch.unsqueeze(FiLM_gamma, -1)
    # repeat entries and go to shape [batch_size, n_channels, group_dim, height, width]
    FiLM_gamma = torch.tile(FiLM_gamma, (group_dim, img_dim, img_dim))

    FiLM_beta = torch.unsqueeze(FiLM_beta, -1)
    FiLM_beta = torch.unsqueeze(FiLM_beta, -1)
    FiLM_beta = torch.unsqueeze(FiLM_beta, -1)
    FiLM_beta = torch.tile(FiLM_beta, (group_dim, img_dim, img_dim))

    res = (1 + FiLM_gamma) * conv_output + FiLM_beta
    expected_res = torch.ones((batch_size, n_channels, group_dim, img_dim, img_dim))
    for i in range(batch_size):
        expected_res[i] = (i + 1) * expected_res[i] + i * expected_res[i]

    assert torch.all(torch.eq(res, expected_res))
    assert res.shape == conv_output.shape
    print(f'res shape: {res.shape}')
    print('great success!')


# c4_test()

