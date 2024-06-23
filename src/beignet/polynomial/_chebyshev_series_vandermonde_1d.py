import operator

import numpy


def chebyshev_series_vandermonde_1d(input, degree):
    ideg = operator.index(degree)

    if ideg < 0:
        raise ValueError

    input = numpy.array(input, ndmin=1) + 0.0

    dims = (ideg + 1,) + input.shape

    dtyp = input.dtype

    v = numpy.empty(dims, dtype=dtyp)

    v[0] = input * 0 + 1

    if ideg > 0:
        x2 = 2 * input
        v[1] = input
        for i in range(2, ideg + 1):
            v[i] = v[i - 1] * x2 - v[i - 2]

    return numpy.moveaxis(v, 0, -1)
