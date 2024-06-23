import torch


def physicists_hermite_series_line(input, other):
    if other != 0:
        return torch.tensor([input, other / 2])
    else:
        return torch.tensor([input])
