import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN
import torch.nn.functional as F
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4

from layers import GFiLM


def CCBN(feature_map_shape: list,
         proj_dim: int,
         group: str,
         specnorm: bool = True,
         initialization: bool = 'orthogonal') -> GFiLM:
    """

    :param feature_map_shape: [no_channels, height, width]
    :param proj_dim: length of noise concatenated with label embedding
    :param group: str, Symmetry group to be equivariant to. One of {'Z2', 'C4', 'D4'}
    :param specnorm: Whether to use spectral normalization on the linear projections.
    :param initialization: Kernel initializer for linear projection.
    :return: FiLM layer
    """
    channels = int(feature_map_shape[0])

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

    return GFiLM(feature_map_shape, x_gamma, x_beta, group=group)


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
        :param h_input: input group, one of {'Z2', 'C4'}
        :param h_output: output group, one of {'Z2', 'C4'}
        :param group_equivariance: whether to be invariant. not yet implemented
        :param kernel_size: convolution kernel size
        :param downsample: whether to downsample 2x
        :param pool: Pooling mode. One of {'avg', 'max'}
        """
        super(DiscBlock, self).__init__()
        self.h_input = h_input
        self.h_output = h_output
        self.group_equivariance = group_equivariance
        self.downsample = downsample
        self.pool = pool

        if group_equivariance is True:
            # TODO orthogonal initialization
            if self.h_input == 'Z2' and self.h_output == 'C4':
                self.conv = SN(P4ConvZ2(in_channels=in_features,
                                        out_channels=out_features,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=kernel_size//2,
                                        bias=True))
            elif self.h_input == 'C4' and h_output == 'C4':
                self.conv = SN(P4ConvP4(in_channels=in_features,
                                        out_channels=out_features,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=kernel_size // 2,
                                        bias=True))
        else:
            self.conv = SN(nn.Conv2d(in_channels=in_features,
                                     out_channels=out_features,
                                     kernel_size=kernel_size,
                                     padding='same',
                                     bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_equivariance is True:
            # x.shape: [batch_size, n_channels, group_dim, height, width]
            out = self.conv(x)
            out = F.leaky_relu_(out, negative_slope=0.2)
            if self.downsample is True:
                s = out.shape
                if self.pool == 'max':
                    out = F.max_pool2d(out.view(s[0], s[1] * s[2], s[3], s[4]), kernel_size=2)
                elif self.pool == 'avg':
                    out = F.avg_pool2d(out.view(s[0], s[1] * s[2], s[3], s[4]), kernel_size=2)
                out = out.view(s[0], s[1], s[2], s[3] // 2, s[4] // 2)
            return out
        else:
            out = self.conv(x)
            out = F.leaky_relu_(out, negative_slope=0.2)
            if self.downsample is True:
                if self.pool == 'max':
                    out = F.max_pool2d(out, kernel_size=2)
                elif self.pool == 'avg':
                    out = F.avg_pool2d(out, kernel_size=2)
        return out
