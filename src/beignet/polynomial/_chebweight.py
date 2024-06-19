import torch


def chebweight(x):
    return 1.0 / (torch.sqrt(1.0 + x) * torch.sqrt(1.0 - x))
