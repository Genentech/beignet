import numpy

from beignet.polynomial import _map_parameters


def _map_domain(x, old, new):
    x = numpy.asanyarray(x)
    off, scl = _map_parameters(old, new)
    return off + scl * x
