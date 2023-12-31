import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils import spectral_norm as SN
from blocks import DiscBlock, DiscBlockDCGAN
from utils import pooling


class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple[int, int, int], disc_arch: str = 'z2_rot_mnist', n_classes: int = 10):
        """
        :param img_shape: tuple [n_channels, height, width]
        :param disc_arch: architecture type.
                One of {'z2_rot_mnist', 'p4_rot_mnist', 'z2_rot_mnist_no_label', 'vanilla', 'vanilla_small'}
        :param n_classes: number of classes, int
        """
        super(Discriminator, self).__init__()
        self.disc_arch = disc_arch
        n_channels = img_shape[0]
        height = img_shape[1]
        width = img_shape[2]
        if 'z2_rot_mnist' in self.disc_arch:
            self.block1 = DiscBlock(in_features=n_channels,
                                    out_features=128,
                                    h_input='Z2',
                                    h_output='Z2',
                                    group_equivariance=False,
                                    pool='avg')
            self.block2 = DiscBlock(in_features=128,
                                    out_features=256,
                                    h_input='Z2',
                                    h_output='Z2',
                                    group_equivariance=False,
                                    pool='avg')
            self.block3 = DiscBlock(in_features=256,
                                    out_features=512,
                                    h_input='Z2',
                                    h_output='Z2',
                                    group_equivariance=False,
                                    pool='avg')
            if self.disc_arch == 'z2_rot_mnist':
                # label embedding layer for projection output
                self.label_emb_linear = nn.Linear(in_features=n_classes,
                                                  out_features=512)

                self.last_layer = SN(nn.Linear(in_features=512,
                                               out_features=1))
            elif self.disc_arch == 'z2_rot_mnist_no_label':
                # omit label embedding layer
                self.last_layer = SN(nn.Linear(in_features=512,
                                               out_features=1))
        elif self.disc_arch == 'p4_rot_mnist':
            self.block1 = DiscBlock(in_features=n_channels,
                                    out_features=64,
                                    h_input='Z2',
                                    h_output='C4',
                                    group_equivariance=True,
                                    pool='avg')
            self.block2 = DiscBlock(in_features=64,
                                    out_features=128,
                                    h_input='C4',
                                    h_output='C4',
                                    group_equivariance=True,
                                    pool='avg')
            self.block3 = DiscBlock(in_features=128,
                                    out_features=256,
                                    h_input='C4',
                                    h_output='C4',
                                    group_equivariance=True,
                                    pool='avg')
            self.label_emb_linear = nn.Linear(in_features=n_classes,
                                              out_features=256)
            self.last_layer = SN(nn.Linear(in_features=256,
                                           out_features=1))
        elif self.disc_arch == 'vanilla':
            features_d = 16
            self.disc = nn.Sequential(
                nn.Conv2d(img_shape[0], features_d, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                DiscBlockDCGAN(features_d, features_d * 2, 4, 2, 1),
                DiscBlockDCGAN(features_d * 2, features_d * 4, 4, 2, 1),
                DiscBlockDCGAN(features_d * 4, features_d * 8, 4, 2, 1),
                nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            )
        elif self.disc_arch == 'vanilla_small':
            features_d = 16
            self.disc = nn.Sequential(
                nn.Conv2d(img_shape[0], features_d, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                DiscBlockDCGAN(features_d, features_d * 2, 4, 2, 1),
                DiscBlockDCGAN(features_d * 2, features_d * 4, 4, 2, 1),
                # DiscBlockDCGAN(features_d * 4, features_d * 8, 4, 2, 1),
                nn.Conv2d(features_d * 4, 1, kernel_size=4, stride=2, padding=1),
            )

    def forward(self, x: list[torch.Tensor, torch.Tensor]):
        """
        :param x: list of [images, labels]
                    images.shape: [batch_size, n_channels, height, width]
                    labels.shape: [batch_size, n_classes]
        :return: forward pass of discriminator
        """
        if 'z2_rot_mnist' in self.disc_arch:
            fea = self.block1(x[0])
            # fea: [batch_size, 128, 14, 14]
            fea = self.block2(fea)
            # fea: [batch_size, 256, 7, 7]
            fea = self.block3(fea)
            # fea: [batch_size, 512, 3, 3]
            flat = F.avg_pool2d(fea, kernel_size=fea.shape[-1])

        elif self.disc_arch == 'p4_rot_mnist':
            fea = self.block1(x[0])
            fea = self.block2(fea)
            fea = self.block3(fea)
            fea = pooling.group_max_pool(fea, 'C4')
            # now fea.shape: [batch_size, n_channels, height, width]
            flat = F.avg_pool2d(fea, kernel_size=fea.shape[-1])

        elif self.disc_arch == 'vanilla' or self.disc_arch == 'vanilla_small':
            return self.disc(x[0])

        flat = torch.squeeze(flat)
        # flat: [batch_size, n_filters_last_block]
        if self.disc_arch == 'z2_rot_mnist_no_label':
            # no labels: omit projection with embedded labels
            prediction = self.last_layer(flat)
            prediction = torch.squeeze(prediction)
        else:
            label_emb = self.label_emb_linear(x[1])
            # label_emb: [batch_size, n_filters_last_block]

            projection = (flat * label_emb).sum(axis=1)
            original_pred = self.last_layer(flat)
            original_pred = torch.squeeze(original_pred)
            prediction = torch.add(projection, original_pred)
            # prediction: [batch_size,]
        return prediction
