import torch


def get_device(debug: bool = False) -> torch.device:
    """
    Get device for torch. One of {'cpu', 'mps', 'cuda'}
    If debug is True, the device is set to 'cpu'
    :param debug: If true, the device will be set to 'cpu'
    :return: torch.device
    """
    if debug is True:
        return torch.device('cpu')
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    return torch.device(device)
