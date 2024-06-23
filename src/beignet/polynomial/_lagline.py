import torch


def laguerre_series_line(input, other):
    if other != 0:
        return torch.tensor([input + other, -other])
    else:
        return torch.tensor([input])
