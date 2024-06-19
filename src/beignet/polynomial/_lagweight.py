import torch


def lagweight(x):
    return torch.exp(-x)
