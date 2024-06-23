import torch


def chebyshev_series_weight(x):
    return 1.0 / (torch.sqrt(1.0 + x) * torch.sqrt(1.0 - x))
