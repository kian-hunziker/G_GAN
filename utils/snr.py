import torch


def snr(tar, approx):
    """
    Signal-to-Noise Ration as 10 * log10(signal_power / noise_power)
    :param tar: Target images, [batch_size, channels, H, W]
    :param approx: Approximations of the target images, [batch_size, channels, H, W]
    :return: snr in dB
    """
    assert tar.shape == approx.shape
    p_signal = torch.sum(tar ** 2, dim=(1, 2, 3))
    p_noise = torch.sum((tar - approx) ** 2, dim=(1, 2, 3))
    return 10 * torch.log10(p_signal / p_noise)


def psnr(tar, approx, data_range: tuple[float, float] = (0, 1)):
    """
    Peak SNR as 10 * log10(R^2 / MSE)
    :param tar: Target images, [batch_size, channels, H, W]
    :param approx: Approximations of the targets, [batch_size, channels, H, W]
    :param data_range: Range of the pixel values (low, high)
    :return: peak snr in dB
    """
    assert tar.shape == approx.shape
    r = data_range[1] - data_range[0]
    mse = torch.mean((tar - approx) ** 2, dim=(1, 2, 3))
    return 10 * torch.log10(r ** 2 / mse)