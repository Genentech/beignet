import numpy
import torch
from torch import Tensor


def evaluate_power_series_1d(x: Tensor, c: Tensor, tensor: bool = True) -> Tensor:
    c = torch.ravel(c)

    if tensor:
        c = numpy.reshape(c, c.shape + (1,) * x.ndim)

    c0 = c[-1] + x * 0

    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0 * x

    return c0
