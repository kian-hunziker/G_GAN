import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, input_shape: list):
        super(FiLM, self).__init__()
        assert isinstance(input_shape, list)
        feature_map_shape = input_shape[0]
        FiLM_gamma_shape = input_shape[1].weight.shape
        FiLM_beta_shape = input_shape[2].weight.shape

        self.gamma_linear = input_shape[1]
        self.beta_linear = input_shape[2]

        self.height = feature_map_shape[1]
        self.width = feature_map_shape[2]
        self.n_feature_maps = feature_map_shape[0]

        assert (int(self.n_feature_maps) == FiLM_gamma_shape[0])
        assert (int(self.n_feature_maps) == FiLM_beta_shape[0])

    def forward(self, x: list) -> torch.Tensor:
        assert isinstance(x, list)
        conv_output, cla = x

        FiLM_gamma = self.gamma_linear(cla)
        FiLM_beta = self.beta_linear(cla)
        '''
        conv_output: [batch_size, channels, height, width]
        FiLM_gamma: [batch_size, channels]
        FiLM_beta: [batch_size, channels]
        '''
        FiLM_gamma = torch.unsqueeze(FiLM_gamma, -1)
        FiLM_gamma = torch.unsqueeze(FiLM_gamma, -1)
        FiLM_gamma = torch.tile(FiLM_gamma, (self.height, self.width))

        FiLM_beta = torch.unsqueeze(FiLM_beta, -1)
        FiLM_beta = torch.unsqueeze(FiLM_beta, -1)
        FiLM_beta = torch.tile(FiLM_beta, (self.height, self.width))

        # apply affine transformation
        return (1 + FiLM_gamma) * conv_output + FiLM_beta
