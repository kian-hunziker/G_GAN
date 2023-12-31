import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN
from blocks import CCBN, GenBlockDCGAN, ResBlockGen
from layers import GFiLM
from utils import pooling, g_batch_norm

try:
    from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4
except ImportError:
    from utils.groupy_dummie import P4ConvZ2, P4ConvP4


class Generator(nn.Module):
    def __init__(self, n_classes: int = 10, gen_arch: str = 'z2_rot_mnist', latent_dim: int = 64):
        """
        Constructor for a generator
        :param n_classes: number of classes for conditional use
        :param gen_arch: Architecture type.
                One of {'z2_rot_mnist', 'p4_rot_mnist', 'z2_rot_mnist_no_label', 'vanilla', 'vanilla_small'}
        :param latent_dim: Dimensionality of latent noise input
        """
        super(Generator, self).__init__()
        self.gen_arch = gen_arch
        self.channel_scale_factor = 1
        self.proj_dim = 128 * 7 * 7
        # changed from original to match pytorch standard
        # [batch_size, n_channels, width, height]
        # used to be (7, 7, 128)
        self.proj_shape = (128, 7, 7)
        self.label_emb_dim = 64

        if gen_arch == 'z2_rot_mnist_no_label':
            self.projected_noise_layer = SN(nn.Linear(in_features=latent_dim,
                                                      out_features=self.proj_dim))
        else:
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
                                      momentum=0.1,
                                      affine=False)
            self.ccbn1 = CCBN(feature_map_shape=[256, 14, 14],
                              proj_dim=self.proj_dim,
                              group='Z2')
            self.conv3 = SN(nn.Conv2d(in_channels=256,
                                      out_channels=128,
                                      kernel_size=3,
                                      padding='same',
                                      bias=False))
            # affine= false reduces no trainable params from 7694208 to 7693440
            self.BN2 = nn.BatchNorm2d(num_features=128,
                                      momentum=0.1,
                                      affine=False)
            # CCBN
            self.ccbn2 = CCBN(feature_map_shape=[128, 28, 28],
                              proj_dim=self.proj_dim,
                              group='Z2')
            self.conv4 = SN(nn.Conv2d(in_channels=128,
                                      out_channels=1,
                                      kernel_size=3,
                                      padding='same',
                                      bias=False))
        elif self.gen_arch == 'z2_rot_mnist_no_label':
            # same as z2_rot_mnist but without CCBN layers. Instead, we set affine=True for BN layers
            # so we should still learn an affine transformation, but it is not dependent on noise input or labels
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
                                      momentum=0.1,
                                      affine=True)
            self.conv3 = SN(nn.Conv2d(in_channels=256,
                                      out_channels=128,
                                      kernel_size=3,
                                      padding='same',
                                      bias=False))
            self.BN2 = nn.BatchNorm2d(num_features=128,
                                      momentum=0.1,
                                      affine=True)
            self.conv4 = SN(nn.Conv2d(in_channels=128,
                                      out_channels=1,
                                      kernel_size=3,
                                      padding='same',
                                      bias=False))
        elif self.gen_arch == 'p4_rot_mnist':
            # TODO: orthogonal initialization for gconv?
            self.conv1 = SN(P4ConvZ2(in_channels=128,
                                     out_channels=256,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=True))
            self.conv2 = SN(P4ConvP4(in_channels=256,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=False))
            self.gbn1 = g_batch_norm.GBatchNorm('C4', num_channels=128, affine=False)
            self.ccbn1 = CCBN(feature_map_shape=[128, 14, 14],
                              proj_dim=self.proj_dim,
                              group='C4')
            self.conv3 = SN(P4ConvP4(in_channels=128,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=False))
            self.gbn2 = g_batch_norm.GBatchNorm('C4', num_channels=64, affine=False)
            self.ccbn2 = CCBN(feature_map_shape=[64, 28, 28],
                              proj_dim=self.proj_dim,
                              group='C4')
            self.conv4 = SN(P4ConvP4(in_channels=64,
                                     out_channels=1,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=False))
        elif self.gen_arch == 'vanilla':
            features_g = 16
            self.gen = nn.Sequential(
                GenBlockDCGAN(latent_dim, features_g * 16, 4, 1, 0),
                GenBlockDCGAN(features_g * 16, features_g * 8, 4, 2, 1),
                GenBlockDCGAN(features_g * 8, features_g * 4, 4, 2, 1),
                GenBlockDCGAN(features_g * 4, features_g * 2, 4, 2, 1),
                nn.ConvTranspose2d(
                    features_g * 2,
                    out_channels=1,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.Tanh()  # [-1, 1]
            )
        elif self.gen_arch == 'vanilla_small':
            features_g = 16
            self.gen = nn.Sequential(
                GenBlockDCGAN(latent_dim, 256, 4, 1, 0),
                GenBlockDCGAN(256, 128, 4, 2, 1),
                GenBlockDCGAN(128, 64, 4, 2, 1),
                GenBlockDCGAN(64, 32, 4, 2, 1),
                # only change these dimensions to generate images of dimension [28, 28]
                nn.ConvTranspose2d(
                    32,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=2
                ),
                nn.Tanh()  # [-1, 1]
            )
        elif self.gen_arch == 'z2_rot_mnist_resblock':
            self.block1 = ResBlockGen(in_features=128,
                                      out_features=512,
                                      input_img_size=7,
                                      projection_dim=self.proj_dim,
                                      h_input='Z2',
                                      h_output='Z2',
                                      pad='same',
                                      stride=1,
                                      group_equiv=False,
                                      kernel_size=3,
                                      upsample=True
                                      )
            self.block2 = ResBlockGen(in_features=512,
                                      out_features=256,
                                      input_img_size=14,
                                      projection_dim=self.proj_dim,
                                      h_input='Z2',
                                      h_output='Z2',
                                      pad='same',
                                      stride=1,
                                      group_equiv=False,
                                      kernel_size=3,
                                      upsample=True
                                      )
            self.block3 = ResBlockGen(in_features=256,
                                      out_features=128,
                                      input_img_size=28,
                                      projection_dim=self.proj_dim,
                                      h_input='Z2',
                                      h_output='Z2',
                                      pad='same',
                                      stride=1,
                                      group_equiv=False,
                                      kernel_size=3,
                                      upsample=False
                                      )
            self.final_conv = SN(nn.Conv2d(in_channels=128,
                                           out_channels=1,
                                           kernel_size=3,
                                           stride=1,
                                           padding='same',
                                           bias=False))

    def forward(self, latent_noise: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        """
        cla: concatenated vector [latent_noise, embedded_labels]
        gen: reshaped input to convolutional layers: [batch_size, (proj_shape)]
        for Z2 gen has shape [batch_size, 128, 7, 7]

        :param latent_noise: noise to synthesize images from [batch_size, latent_dim]
        :param label: class labels one hot encoded [batch_size, n_classes]
        :return: synthesized images [batch_size, n_channels, height, width]
        """

        if self.gen_arch == 'z2_rot_mnist_no_label':
            # cla: projected noise, as a vector of length proj_dim
            cla = self.projected_noise_layer(latent_noise)
            # gen: input for convolutions, shape: proj_shape
            gen = cla.reshape(tuple([-1]) + self.proj_shape)
        elif self.gen_arch == 'z2_rot_mnist' or self.gen_arch == 'p4_rot_mnist':
            label_projection = self.label_projection_layer(label)
            cla = torch.cat((latent_noise, label_projection), dim=1)
            cla = self.projected_noise_and_classes(cla)
            # TODO make reshape nicer
            gen = cla.reshape(tuple([-1]) + self.proj_shape)
        if self.gen_arch == 'z2_rot_mnist':
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
        elif self.gen_arch == 'z2_rot_mnist_no_label':
            fea = F.relu(self.conv1(gen))
            fea = F.interpolate(fea, scale_factor=2)
            # [batch_size, 512, 14, 14]
            fea = F.relu(self.BN1(self.conv2(fea)))
            # [batch_size, 256, 14, 14]
            fea = F.interpolate(fea, scale_factor=2)
            fea = F.relu(self.BN2(self.conv3(fea)))
            fea = self.conv4(fea)
            # [batch_size, 1, 28, 28]
        elif self.gen_arch == 'p4_rot_mnist':
            fea = F.relu(self.conv1(gen))

            # upsample 2x
            s = fea.shape
            fea = F.interpolate(fea.view(s[0], s[1] * s[2], s[3], s[4]), scale_factor=2)
            fea = fea.view(s[0], s[1], s[2], 2 * s[3], 2 * s[4])
            fea = F.relu(self.ccbn1([self.gbn1(self.conv2(fea)), cla]))

            # upsample 2x
            s = fea.shape
            fea = F.interpolate(fea.view(s[0], s[1] * s[2], s[3], s[4]), scale_factor=2)
            fea = fea.view(s[0], s[1], s[2], 2 * s[3], 2 * s[4])

            fea = F.relu(self.ccbn2([self.gbn2(self.conv3(fea)), cla]))

            fea = self.conv4(fea)
            fea = pooling.group_max_pool(fea, group='C4')
        elif self.gen_arch == 'vanilla' or self.gen_arch == 'vanilla_small':
            return self.gen(latent_noise.unsqueeze(-1).unsqueeze(-1))
        elif self.gen_arch == 'z2_rot_mnist_resblock':
            label_projection = self.label_projection_layer(label)
            cla = torch.cat((latent_noise, label_projection), dim=1)
            cla = self.projected_noise_and_classes(cla)
            # TODO make reshape nicer
            gen = cla.reshape(tuple([-1]) + self.proj_shape)

            fea = self.block1(gen, cla)
            fea = self.block2(fea, cla)
            fea = self.block3(fea, cla)
            fea = self.final_conv(fea)

        out = F.tanh(fea)
        return out


def initialize_weights(model: nn.Module, arch: str, show_details: bool = False):
    if arch == 'vanilla' or arch == 'vanilla_small' or arch == 'z2_rot_mnist_no_label':
        print(f'  ->  Initializing from normal distribution')
        # initialize weights according to DCGAN paper
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
    else:
        # initialize weights orthogonally according to gGan paper
        print(f'  ->  Orthogonal initialization')
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if show_details is True:
                    print(f'initialized {m}')
            elif isinstance(m, GFiLM):
                nn.init.orthogonal_(m.beta_linear.weight)
                nn.init.orthogonal_(m.gamma_linear.weight)
                if show_details is True:
                    print(f'initialized film layer {m}')
