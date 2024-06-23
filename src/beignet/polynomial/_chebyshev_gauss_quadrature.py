import operator

import torch
from torch import Tensor


def chebyshev_gauss_quadrature(input: Tensor) -> (Tensor, Tensor):
    degree = operator.index(input)

    if degree <= 0:
        raise ValueError

    output = torch.arange(1, 2 * degree, 2) / (2.0 * degree)

    output = output * torch.pi

    output = torch.cos(output)

    weight = torch.ones(degree) * (torch.pi / degree)

    return output, weight
