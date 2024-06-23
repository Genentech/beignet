import torch


def physicists_hermite_series_weight(x):
    return torch.exp(-(x**2))
