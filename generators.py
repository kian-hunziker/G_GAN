import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN
from blocks import CCBN
from layers import GFiLM


class Generator(nn.Module):
    def __init__(self, n_classes: int = 10, gen_arch: str = 'z2_rot_mnist', latent_dim: int = 64):
        super(Generator, self).__init__()
        self.gen_arch = gen_arch
        self.channel_scale_factor = 1
        self.proj_dim = 128 * 7 * 7
        # changed from original to match pytorch standard
        # [batch_size, n_channels, width, height]
        # used to be (7, 7, 128)
        self.proj_shape = (128, 7, 7)
        self.label_emb_dim = 64

        self.label_projection_layer = nn.Linear(in_features=n_classes,
                                                out_features=self.label_emb_dim,
                                                bias=False)
        self.projected_noise_and_classes = SN(nn.Linear(in_features=latent_dim + self.label_emb_dim,
                                                        out_features=self.proj_dim))

        if self.gen_arch == 'z2_rot_mnist':
            self.conv1 = SN(nn.Conv2d(in_channels=128,
                                      out_channels=512,
                                      kernel_size=3,
                                      padding='same',
                                      bias=True))
            self.conv2 = SN(nn.Conv2d(in_channels=512,
                                      out_channels=256,
                                      kernel_size=3,
                                      padding='same',
                                      bias=False))
            self.BN1 = nn.BatchNorm2d(num_features=256,
                                      momentum=0.1)
            self.ccbn1 = CCBN(feature_map_shape=[256, 14, 14],
                              proj_dim=self.proj_dim,
                              group='Z2')
            self.conv3 = SN(nn.Conv2d(in_channels=256,
                                      out_channels=128,
                                      kernel_size=3,
                                      padding='same',
                                      bias=False))
            self.BN2 = nn.BatchNorm2d(num_features=128,
                                      momentum=0.1)
            # CCBN
            self.ccbn2 = CCBN(feature_map_shape=[128, 28, 28],
                              proj_dim=self.proj_dim,
                              group='Z2')
            self.conv4 = SN(nn.Conv2d(in_channels=128,
                                      out_channels=1,
                                      kernel_size=3,
                                      padding='same',
                                      bias=False))

    def forward(self, latent_noise: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        '''
        cla: concatenated vector [latent_noise, embedded_labels]
        gen: reshaped input to convolutional layers: [batch_size, (proj_shape)]
        for Z2 gen has shape [batch_size, 128, 7, 7]

        :param latent_noise: noise to synthesize images from [batch_size, latent_dim]
        :param label: class labels one hot encoded [batch_size, n_classes]
        :return: synthesized images [batch_size, n_channels, height, width]
        '''
        if self.gen_arch == 'z2_rot_mnist':
            label_projection = self.label_projection_layer(label)
            cla = torch.cat((latent_noise, label_projection), dim=1)
            cla = self.projected_noise_and_classes(cla)
            # TODO make reshape nicer
            gen = cla.reshape(tuple([-1]) + self.proj_shape)
            # now cla should be: [batch_size,128, 7, 7]
            fea = self.conv1(gen)
            fea = F.relu(fea)
            fea = F.interpolate(fea, scale_factor=2)
            # [batch_size, 512, 14, 14]
            fea = self.conv2(fea)
            # [batch_size, 256, 14, 14]
            fea = self.BN1(fea)
            # CCBN
            fea = self.ccbn1([fea, cla])
            fea = F.relu(fea)
            fea = F.interpolate(fea, scale_factor=2)
            fea = self.conv3(fea)
            # [batch_size, 128, 28, 28]
            fea = self.BN2(fea)
            # CCBN
            fea = self.ccbn2([fea, cla])
            fea = F.relu(fea)
            fea = self.conv4(fea)
            # [batch_size, 1, 28, 28

        out = F.tanh(fea)
        return out


def init_generator_weights_z2(m, show_details=False):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        if show_details is True:
            print(f'initialized {m}')
    elif isinstance(m, GFiLM):
        nn.init.orthogonal_(m.beta_linear.weight)
        nn.init.orthogonal_(m.gamma_linear.weight)
        if show_details is True:
            print(f'initialized film layer {m}')


def generator_debug_test():
    gen = Generator()
    batch_size = 32
    z_dim = 64
    n_classes = 10
    labels = torch.zeros((batch_size, n_classes))
    noise = torch.randn((batch_size, z_dim))

    out = gen(noise, labels)
    print(out.shape)


def generator_initialisation_test():
    gen = Generator()
    gen.apply(init_generator_weights_z2)
    print('great success!')

# generator_initialisation_test()
