import numpy


def _nth_slice(i, ndim):
    sl = [numpy.newaxis] * ndim
    sl[i] = slice(None)
    return tuple(sl)
