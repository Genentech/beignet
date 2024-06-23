import numpy

from .__as_series import _as_series
from ._physicists_hermite_series_companion import physicists_hermite_series_companion


def physicists_hermite_series_roots(input):
    [input] = _as_series([input])
    if len(input) <= 1:
        return numpy.array([], dtype=input.dtype)
    if len(input) == 2:
        return numpy.array([-0.5 * input[0] / input[1]])

    m = physicists_hermite_series_companion(input)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r
