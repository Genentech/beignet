import torch


def hermline(input, other):
    if other != 0:
        return torch.tensor([input, other / 2])
    else:
        return torch.tensor([input])
