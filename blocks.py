import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN
import torch.nn.functional as F

from layers import FiLM


def CCBN(feature_map_shape: list,
         proj_dim: int,
         group: str,
         specnorm: bool = True,
         initialization: bool = 'orthogonal') -> FiLM:
    """

    :param feature_map_shape: [no_channels, height, width]
    :param proj_dim: length of noise concatenated with label embedding
    :param group: str, Symmetry group to be equivariant to. One of {'Z2', 'C4', 'D4'}
    :param specnorm: Whether to use spectral normalization on the linear projections.
    :param initialization: Kernel initializer for linear projection.
    :return: FiLM layer
    """
    channels = feature_map_shape[0]
    if group == 'Z2':
        channels = int(channels // 1)

    if specnorm is True:
        x_beta = SN(nn.Linear(in_features=proj_dim,
                              out_features=channels))
        x_gamma = SN(nn.Linear(in_features=proj_dim,
                               out_features=channels))
    else:
        x_beta = nn.Linear(in_features=proj_dim,
                           out_features=channels)
        x_gamma = nn.Linear(in_features=proj_dim,
                            out_features=channels)

    if group == 'Z2':
        return FiLM([feature_map_shape, x_gamma, x_beta])


class DiscBlock(nn.Module):
    def __init__(self, in_features: int,
                 out_features: int,
                 h_input: str,
                 h_output: str,
                 group_equivariance: bool = False,
                 kernel_size: int = 3,
                 downsample: bool = True,
                 pool: str = 'max'):
        """
        :param in_features: number of input channels for conv2d
        :param out_features: number of out channels of conv2d
        :param h_input: input group, one of {'Z2'}
        :param h_output: output group, one of {'Z2'}
        :param group_equivariance: whether to be invariant. not yet implemented
        :param kernel_size: convolution kernel size
        :param downsample: whether to downsample 2x
        :param pool: Pooling mode. One of {'avg', 'max'}
        """
        super(DiscBlock, self).__init__()
        self.downsample = downsample
        self.pool = pool

        if group_equivariance is True:
            return
        else:
            self.conv = SN(nn.Conv2d(in_channels=in_features,
                                     out_channels=out_features,
                                     kernel_size=kernel_size,
                                     padding='same',
                                     bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = F.leaky_relu_(out, negative_slope=0.2)
        if self.downsample is True:
            if self.pool == 'max':
                out = F.max_pool2d(out, kernel_size=2)
            elif self.pool == 'avg':
                out = F.avg_pool2d(out, kernel_size=2)
        return out
