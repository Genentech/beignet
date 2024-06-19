import numpy

from .__pow import _pow


def polypow(c, pow, maxpower=None):
    return _pow(numpy.convolve, c, pow, maxpower)
