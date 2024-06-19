import operator

import numpy


def chebgauss(input):
    ideg = operator.index(input)

    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    output = numpy.arange(1, 2 * ideg, 2) / (2.0 * ideg)

    output = output * numpy.pi

    output = numpy.cos(output)

    weight = numpy.ones(ideg) * (numpy.pi / ideg)

    return output, weight
