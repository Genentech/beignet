import numpy

from .__deprecate_as_int import _deprecate_as_int
from ._legcompanion import legcompanion
from ._legder import legder
from ._legval import legval


def leggauss(deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * deg + [1])
    m = legcompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = legval(x, c)
    df = legval(x, legder(c))
    x -= dy / df

    fm = legval(x, c[1:])
    fm /= numpy.abs(fm).max()
    df /= numpy.abs(df).max()
    w = 1 / (fm * df)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w *= 2.0 / w.sum()

    return x, w
