import numpy

from beignet.polynomial._polycompanion import polycompanion

from .__as_series import _as_series


def polyroots(series):
    (series,) = _as_series([series])

    if len(series) < 2:
        return numpy.array([], dtype=series.dtype)

    if len(series) == 2:
        return numpy.array([-series[0] / series[1]])

    output = polycompanion(series)

    output = numpy.flip(output, axis=0)
    output = numpy.flip(output, axis=1)

    output = numpy.linalg.eigvals(output)

    output = numpy.sort(output)

    return output
