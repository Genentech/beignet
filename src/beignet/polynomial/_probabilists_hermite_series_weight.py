import torch


def probabilists_hermite_series_weight(x):
    return torch.exp(-0.5 * x**2)
