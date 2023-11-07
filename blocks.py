import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN
import torch.nn.functional as F

try:
    from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4
except ImportError:
    from utils.groupy_dummie import P4ConvZ2, P4ConvP4

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
                                        padding=kernel_size // 2,
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


class GenBlockDCGAN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(GenBlockDCGAN, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class DiscBlockDCGAN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DiscBlockDCGAN, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        return self.block(x)


class ResBlockGen(nn.Module):
    def __init__(self, in_features, out_features, input_img_size, projection_dim, h_input='z2', h_output='z2',
                 pad=0, stride=1, group_equiv=False, kernel_size=3, bn_eps=1e-3, upsample=True):

        super(ResBlockGen, self).__init__()
        feature_map_shape_input = [in_features, input_img_size, input_img_size]
        self.upsample = upsample
        if upsample is True:
            feature_map_shape_2 = [out_features, 2 * input_img_size, 2 * input_img_size]
        else:
            feature_map_shape_2 = [out_features, input_img_size, input_img_size]

        self.BN1 = nn.BatchNorm2d(num_features=in_features,
                                  momentum=0.1,
                                  affine=False)
        self.CCBN1 = CCBN(feature_map_shape=feature_map_shape_input,
                          proj_dim=projection_dim,
                          group='Z2')
        self.conv1 = SN(nn.Conv2d(in_channels=in_features,
                                  out_channels=out_features,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=pad,
                                  bias=False))
        self.BN2 = nn.BatchNorm2d(num_features=out_features,
                                  momentum=0.1,
                                  affine=False)
        self.CCBN2 = CCBN(feature_map_shape=feature_map_shape_2,
                          proj_dim=projection_dim,
                          group='Z2')
        self.conv2 = SN(nn.Conv2d(in_channels=out_features,
                                  out_channels=out_features,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=pad,
                                  bias=False))

        self.shortcut_conv = SN(nn.Conv2d(in_channels=in_features,
                                          out_channels=out_features,
                                          kernel_size=1,
                                          stride=stride,
                                          padding=pad,
                                          bias=False))

    def forward(self, x, cla):
        if self.upsample is True:
            shortcut = F.interpolate(x, scale_factor=2)
            shortcut = self.shortcut_conv(shortcut)
        else:
            shortcut = self.shortcut_conv(x)

        x_conv = self.BN1(x)
        x_conv = self.CCBN1([x_conv, cla])
        x_conv = F.relu(x_conv)
        if self.upsample is True:
            x_conv = F.interpolate(x_conv, scale_factor=2)
        x_conv = self.conv1(x_conv)
        x_conv = self.BN2(x_conv)
        x_conv = self.CCBN2([x_conv, cla])
        x_conv = F.relu(x_conv)
        x_conv = self.conv2(x_conv)

        out = x_conv + shortcut
        return out




