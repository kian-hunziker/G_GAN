import torch.nn as nn
from torch.nn.utils import spectral_norm as SN
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