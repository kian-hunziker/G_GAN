import collections

collections.Callable = collections.abc.Callable
import unittest
import torch

import generators
import discriminators


class TestGenerators(unittest.TestCase):

    def setUp(self) -> None:
        self.latent_dim = 64
        self.n_classes = 10
        self.batch_size = 32
        self.img_size = 28
        self.n_channels = 1

        self.labels = torch.zeros(self.batch_size, self.n_classes)
        self.noise = torch.randn(self.batch_size, self.latent_dim)

        self.gen_archs = ['z2_rot_mnist', 'p4_rot_mnist', 'vanilla_small']

    def get_generator(self, gen_arch):
        return generators.Generator(n_classes=self.n_classes,
                                    gen_arch=gen_arch,
                                    latent_dim=self.latent_dim)

    def test_z2_rot_mnist_setup(self):
        gen_arch = 'z2_rot_mnist'
        gen = self.get_generator(gen_arch)
        out = gen(self.noise, self.labels)
        self.assertEqual(out.shape[0], self.batch_size)
        self.assertEqual(out.shape[1], self.n_channels)
        self.assertEqual(out.shape[2], self.img_size)
        self.assertEqual(out.shape[2], self.img_size)

    def test_p4_rot_mnist_setup(self):
        gen_arch = 'p4_rot_mnist'
        gen = self.get_generator(gen_arch)
        out = gen(self.noise, self.labels)
        self.assertEqual(out.shape[0], self.batch_size)
        self.assertEqual(out.shape[1], self.n_channels)
        self.assertEqual(out.shape[2], self.img_size)
        self.assertEqual(out.shape[2], self.img_size)

    def test_vanilla_small_setup(self):
        gen_arch = 'vanilla_small'
        gen = self.get_generator(gen_arch)
        out = gen(self.noise, None)
        self.assertEqual(out.shape[0], self.batch_size)
        self.assertEqual(out.shape[1], self.n_channels)
        self.assertEqual(out.shape[2], self.img_size)
        self.assertEqual(out.shape[2], self.img_size)

    def test_weight_initialization(self):
        for arch in self.gen_archs:
            gen = self.get_generator(arch)
            print(f'init {arch} generator weights')
            generators.initialize_weights(gen, arch)


class TestDiscriminator(unittest.TestCase):
    def setUp(self) -> None:
        self.latent_dim = 64
        self.n_classes = 10
        self.batch_size = 32
        self.img_size = 28
        self.n_channels = 1

        self.labels = torch.zeros(self.batch_size, self.n_classes)
        self.images = torch.randn(self.batch_size, self.n_channels, self.img_size, self.img_size)

        self.disc_archs = ['z2_rot_mnist', 'p4_rot_mnist', 'vanilla_small']

    def get_discriminator(self, disc_arch):
        img_shape = (self.n_channels, self.img_size, self.img_size)
        return discriminators.Discriminator(img_shape=img_shape,
                                            disc_arch=disc_arch,
                                            n_classes=self.n_classes)

    def test_z2_rot_mnist_setup(self):
        disc_arch = 'z2_rot_mnist'
        disc = self.get_discriminator(disc_arch=disc_arch)
        out = disc([self.images, self.labels])
        self.assertEqual(out.shape[0], self.batch_size)

    def test_p4_rot_mnist_setup(self):
        disc_arch = 'p4_rot_mnist'
        disc = self.get_discriminator(disc_arch=disc_arch)
        out = disc([self.images, self.labels])
        self.assertEqual(out.shape[0], self.batch_size)

    def test_vanilla_small_setup(self):
        disc_arch = 'vanilla_small'
        disc = self.get_discriminator(disc_arch=disc_arch)
        out = disc([self.images, self.labels])
        self.assertEqual(out.shape[0], self.batch_size)

    def test_weight_initialization(self):
        for arch in self.disc_archs:
            disc = self.get_discriminator(arch)
            print(f'init {arch} discriminator weights')
            generators.initialize_weights(disc, arch)


if __name__ == '__main__':
    unittest.main()
