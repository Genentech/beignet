import numpy
import torch

from beignet.polynomial import _as_series, _c_series_to_z_series


def chebmul(input, other):
    [input, other] = _as_series([input, other])

    input = _c_series_to_z_series(input)
    other = _c_series_to_z_series(other)

    output = numpy.convolve(input, other)

    n = (output.size + 1) // 2

    output = output[n - 1 :]

    output[1:n] = output[1:n] * 2

    if len(output) != 0 and output[-1] == 0:
        for index in range(len(output) - 1, -1, -1):
            if output[index] != 0:
                break

        output = output[: index + 1]

    output = torch.from_numpy(output)

    return output
