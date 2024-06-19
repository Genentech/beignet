import operator

import numpy

from beignet.polynomial import _normed_hermite_e_n, hermecompanion


def hermegauss(input):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * input + [1])
    m = hermecompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = _normed_hermite_e_n(x, ideg)
    df = _normed_hermite_e_n(x, ideg - 1) * numpy.sqrt(ideg)
    x -= dy / df

    fm = _normed_hermite_e_n(x, ideg - 1)
    fm /= numpy.abs(fm).max()
    w = 1 / (fm * fm)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w *= numpy.sqrt(2 * numpy.pi) / w.sum()

    return x, w
