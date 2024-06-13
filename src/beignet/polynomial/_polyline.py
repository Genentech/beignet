import numpy


def polyline(off, scl):
    if scl != 0:
        return numpy.array([off, scl])
    else:
        return numpy.array([off])
