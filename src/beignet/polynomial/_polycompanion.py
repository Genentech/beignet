import numpy
import torch

from .__as_series import _as_series


def polycompanion(series):
    (series,) = _as_series([series])

    if len(series) < 2:
        raise ValueError

    if len(series) == 2:
        output = numpy.array([[-series[0] / series[1]]])
    else:
        n = series.shape[-1] - 1

        output = torch.zeros([n, n], dtype=series.dtype)

        bot = numpy.reshape(output, -1)[n :: n + 1]

        bot[...] = 1

        output[:, -1] = output[:, -1] - (series[:-1] / series[-1])

    return output
