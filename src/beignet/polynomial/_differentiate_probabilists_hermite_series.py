import operator

import numpy
import torch

from .__normalize_axis_index import _normalize_axis_index


def differentiate_probabilists_hermite_series(input, m=1, scl=1, axis=0):
    input = torch.ravel(input)

    # if input.dtype.char in "?bBhHiIlLqQpP":
    #     input = input.astype(numpy.double)

    cnt = operator.index(m)

    iaxis = operator.index(axis)

    if cnt < 0:
        raise ValueError

    iaxis = _normalize_axis_index(iaxis, input.ndim)

    if cnt == 0:
        return input

    input = numpy.moveaxis(input, iaxis, 0)

    n = len(input)

    if cnt >= n:
        return input[:1] * 0
    else:
        for _ in range(cnt):
            n = n - 1

            input *= scl

            der = numpy.empty((n,) + input.shape[1:], dtype=input.dtype)

            for j in range(n, 0, -1):
                der[j - 1] = j * input[j]

            input = der

    input = numpy.moveaxis(input, 0, iaxis)

    return input
