import torch


def laguerre_series_weight(x):
    return torch.exp(-x)
