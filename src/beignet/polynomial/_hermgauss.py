import operator

import numpy

from beignet.polynomial import _normed_hermite_n, hermcompanion


def hermgauss(input):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * input + [1], dtype=numpy.float64)
    m = hermcompanion(c)
    output = numpy.linalg.eigvalsh(m)

    dy = _normed_hermite_n(output, ideg)
    df = _normed_hermite_n(output, ideg - 1) * numpy.sqrt(2 * ideg)
    output -= dy / df

    fm = _normed_hermite_n(output, ideg - 1)
    fm /= numpy.abs(fm).max()
    weight = 1 / (fm * fm)

    weight = (weight + weight[::-1]) / 2
    output = (output - output[::-1]) / 2

    weight = weight * (numpy.sqrt(numpy.pi) / numpy.sum(weight))

    return output, weight
