import numpy

from ._mapparms import mapparms


def mapdomain(x, old, new):
    x = numpy.asanyarray(x)

    off, scl = mapparms(old, new)

    return off + scl * x
