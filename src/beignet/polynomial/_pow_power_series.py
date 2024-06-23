import numpy

from .__pow import _pow


def pow_power_series(c, pow, maxpower=None):
    return _pow(numpy.convolve, c, pow, maxpower)
