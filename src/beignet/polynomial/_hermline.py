import numpy


def hermline(off, scl):
    if scl != 0:
        return numpy.array([off, scl / 2])
    else:
        return numpy.array([off])
