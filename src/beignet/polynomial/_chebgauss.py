import numpy

from .__deprecate_as_int import _deprecate_as_int


def chebgauss(deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    x = numpy.cos(numpy.pi * numpy.arange(1, 2 * ideg, 2) / (2.0 * ideg))
    w = numpy.ones(ideg) * (numpy.pi / ideg)

    return x, w
