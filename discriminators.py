import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils import spectral_norm as SN
from blocks import DiscBlock
from utils import pooling


class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple[int, int, int], disc_arch: str = 'z2_rot_mnist', n_classes: int = 10):
        """
        :param img_shape: tuple [n_channels, height, width]
        :param disc_arch: architecture type, one of {'z2_rot_mnist', 'p4_rot_mnist'}
        :param n_classes: number of classes, int
        """
        super(Discriminator, self).__init__()
        self.disc_arch = disc_arch
        n_channels = img_shape[0]
        height = img_shape[1]
        width = img_shape[2]
        if self.disc_arch == 'z2_rot_mnist':
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
            # 3 x 3 feature?
            self.label_emb_linear = nn.Linear(in_features=n_classes,
                                              out_features=512)

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

    def forward(self, x: list[torch.Tensor, torch.Tensor]):
        """
        :param x: list of [images, labels]
                    images.shape: [batch_size, n_channels, height, width]
                    labels.shape: [batch_size, n_classes]
        :return: forward pass of discriminator
        """
        if self.disc_arch == 'z2_rot_mnist':
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

        flat = torch.squeeze(flat)
        # flat: [batch_size, n_filters_last_block]
        label_emb = self.label_emb_linear(x[1])
        # label_emb: [batch_size, n_filters_last_block]

        projection = (flat * label_emb).sum(axis=1)
        original_pred = self.last_layer(flat)
        original_pred = torch.squeeze(original_pred)
        prediction = torch.add(projection, original_pred)
        # prediction: [batch_size,]
        return prediction


def init_discriminator_weights_z2(m, show_details=False):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        if show_details is True:
            print(f'initialized {m}')


def discriminator_initialisation_test():
    disc = Discriminator((1, 28, 28), n_classes=10)
    disc.apply(init_discriminator_weights_z2)
    print('great success!')


def discriminator_debug_test(disc_arch):
    img_dim = (1, 28, 28)
    batch_size = 32
    z_dim = 64
    n_classes = 10

    # Generate random one-hot encoded vectors
    one_hot_vectors = []
    for _ in range(batch_size):
        # Randomly select a class (0 to 9)
        class_index = random.randint(0, n_classes - 1)

        # Create a one-hot encoded vector for the selected class
        one_hot_vector = torch.zeros(n_classes)
        one_hot_vector[class_index] = 1.0

        one_hot_vectors.append(one_hot_vector)

    # Convert the list of one-hot vectors to a PyTorch tensor
    labels = torch.stack(one_hot_vectors)
    images = torch.randn(tuple([batch_size]) + img_dim)

    assert (tuple(images.shape) == (batch_size, 1, 28, 28))

    disc = Discriminator(img_shape=img_dim,
                         disc_arch=disc_arch,
                         n_classes=n_classes)

    out = disc([images, labels])
    print(out.shape)

# discriminator_initialisation_test()
# discriminator_debug_test('p4_rot_mnist')
