import numpy

from beignet.polynomial import _as_series, hermecompanion


def hermeroots(input):
    [input] = _as_series([input])
    if len(input) <= 1:
        return numpy.array([], dtype=input.dtype)
    if len(input) == 2:
        return numpy.array([-input[0] / input[1]])

    m = hermecompanion(input)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r
