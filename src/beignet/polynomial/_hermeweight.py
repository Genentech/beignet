import torch


def hermeweight(x):
    return torch.exp(-0.5 * x**2)
