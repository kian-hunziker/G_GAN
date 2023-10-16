import torch


def group_max_pool(x: torch.Tensor, group: str) -> torch.Tensor:
    """
    :param x: input of shape [batch_size, channels, group_layers, height, width]
    :param group: group name, one of {'C4'}
    :return: max value over group layers. out.shape: [batch_size, channels, 1, height, width]
    """
    if group == 'C4':
        # TODO: do we need to do that for channels separately?
        #s = x.shape
        #out = x.view(s[0], s[1] * s[2], s[3], s[4])
        #out = torch.max(out, dim=1)
        #return out.view(s[0], 1, 1, s[3], s[4])
        out = torch.max(x, dim=2)[0]
        out = out.unsqueeze(2)
        return out


def group_max_pool_test_shape():
    batch_size = 64
    n_channels = 3
    img_dim = 28
    group_size = 4
    test_noise = torch.randn((batch_size, n_channels, group_size, img_dim, img_dim))
    out = group_max_pool(test_noise, group='C4')
    assert out.shape == (batch_size, n_channels, 1, img_dim, img_dim)
    print(f"target shape: {(batch_size, n_channels, 1, img_dim, img_dim)}, actual shape: {out.shape}")


# group_max_pool_test_shape()
