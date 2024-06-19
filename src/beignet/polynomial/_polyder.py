import operator

import numpy

from .__normalize_axis_index import _normalize_axis_index


def polyder(input, order: int = 1, scale: float = 1, axis: int = 0):
    output = numpy.array(input, ndmin=1)

    if output.dtype.char in "?bBhHiIlLqQpP":
        output = output + 0.0

    dtype = output.dtype

    cnt = operator.index(order)

    axis = operator.index(axis)

    if cnt < 0:
        raise ValueError

    axis = _normalize_axis_index(axis, output.ndim)

    if cnt == 0:
        return output

    output = numpy.moveaxis(output, axis, 0)

    n = len(output)

    if cnt >= n:
        output = output[:1] * 0
    else:
        for _ in range(cnt):
            n = n - 1

            output = output * scale

            der = numpy.empty((n,) + output.shape[1:], dtype=dtype)

            for j in range(n, 0, -1):
                der[j - 1] = j * output[j]

            output = der

    output = numpy.moveaxis(output, 0, axis)

    return output
