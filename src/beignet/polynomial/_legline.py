import torch


def legendre_series_line(input, other):
    if other != 0:
        return torch.tensor([input, other])
    else:
        return torch.tensor([input])
