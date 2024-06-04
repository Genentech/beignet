import torch

from ._faddeeva_w import faddeeva_w


def erfc(z):
    return torch.exp(-z.pow(2)) * faddeeva_w(1j * z)
