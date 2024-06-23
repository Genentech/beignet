import numpy
import torch


def evaluate_power_series_1d(x, c, tensor=True):
    c = torch.ravel(c)

    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c + 0.0

    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)

    if isinstance(x, numpy.ndarray) and tensor:
        c = numpy.reshape(c, c.shape + (1,) * x.ndim)

    c0 = c[-1] + x * 0

    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0 * x

    return c0
