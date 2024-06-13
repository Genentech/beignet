import numpy


def _cseries_to_zseries(c):
    n = c.size
    zs = numpy.zeros(2 * n - 1, dtype=c.dtype)
    zs[n - 1 :] = c / 2
    return zs + zs[::-1]
