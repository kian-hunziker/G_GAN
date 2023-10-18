import torch
import torch.nn as nn

'''
https://github.com/neel-dey/tf2-keras-gcnn/blob/master/keras_gcnn/layers/normalization.py
'''


class GBatchNorm(nn.GroupNorm):
    # TODO add momentum to g batch norm (moving average)
    def __init__(self, group, num_channels, eps=1e-05, affine=True, device=None, dtype=None):
        self.group = group
        if self.group == 'Z2':
            num_groups = num_channels
        elif self.group == 'C4':
            num_groups = num_channels
            num_channels = num_channels * 4
        super(GBatchNorm, self).__init__(num_groups=num_groups,
                                         num_channels=num_channels,
                                         eps=eps,
                                         affine=affine,
                                         device=device,
                                         dtype=dtype)

    def __call__(self, x):
        if self.group == 'Z2':
            # not really needed, for Z2 we use nn.BatchNorm2d
            return super().__call__(x)
        elif self.group == 'C4':
            # x has shape [batch_size, n_channels, 4, height, width]
            s = x.shape
            x_reshaped = x.view(s[0], s[1] * s[2], s[3], s[4])
            x_norm = super().__call__(x_reshaped)
            return x_norm.view(s[0], s[1], s[2], s[3], s[4])


def batch_norm_basic_test():
    batch_size = 64
    n_channels = 128
    img_dim = 28
    gbn_c4 = GBatchNorm(group='C4',
                        num_channels=n_channels,
                        affine=False)
    gbn_z2 = GBatchNorm(group='Z2',
                        num_channels=128,
                        affine=False)
    test_input_c4 = torch.randn((batch_size, n_channels, 4, img_dim, img_dim))
    test_input_z2 = torch.randn((batch_size, n_channels, img_dim, img_dim))
    out_c4 = gbn_c4(test_input_c4)
    out_z2 = gbn_z2(test_input_z2)
    print(f'out shape: {out_c4.shape}')
    assert out_c4.shape == test_input_c4.shape
    assert out_z2.shape == test_input_z2.shape

# batch_norm_basic_test()
