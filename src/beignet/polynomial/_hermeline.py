import torch


def hermeline(input, other):
    if other != 0:
        return torch.tensor([input, other])
    else:
        return torch.tensor([input])
