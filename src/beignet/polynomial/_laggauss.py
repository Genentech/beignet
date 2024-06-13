import numpy

from .__deprecate_as_int import _deprecate_as_int
from ._lagcompanion import lagcompanion
from ._lagder import lagder
from ._lagval import lagval


def laggauss(deg):
    ideg = _deprecate_as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * deg + [1])
    m = lagcompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = lagval(x, c)
    df = lagval(x, lagder(c))
    x -= dy / df

    fm = lagval(x, c[1:])
    fm /= numpy.abs(fm).max()
    df /= numpy.abs(df).max()
    w = 1 / (fm * df)

    w /= w.sum()

    return x, w
