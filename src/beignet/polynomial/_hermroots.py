import numpy

from .__as_series import _as_series
from ._hermcompanion import hermcompanion


def hermroots(input):
    [input] = _as_series([input])
    if len(input) <= 1:
        return numpy.array([], dtype=input.dtype)
    if len(input) == 2:
        return numpy.array([-0.5 * input[0] / input[1]])

    m = hermcompanion(input)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r
