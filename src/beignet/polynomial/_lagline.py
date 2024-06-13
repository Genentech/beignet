import numpy


def lagline(off, scl):
    if scl != 0:
        return numpy.array([off + scl, -scl])
    else:
        return numpy.array([off])
