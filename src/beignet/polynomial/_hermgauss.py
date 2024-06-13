import numpy

from .__deprecate_as_int import _deprecate_as_int
from .__normed_hermite_n import _normed_hermite_n
from ._hermcompanion import hermcompanion


def hermgauss(deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * deg + [1], dtype=numpy.float64)
    m = hermcompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = _normed_hermite_n(x, ideg)
    df = _normed_hermite_n(x, ideg - 1) * numpy.sqrt(2 * ideg)
    x -= dy / df

    fm = _normed_hermite_n(x, ideg - 1)
    fm /= numpy.abs(fm).max()
    w = 1 / (fm * fm)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w *= numpy.sqrt(numpy.pi) / w.sum()

    return x, w
