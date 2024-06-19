import numpy

from .__map_parameters import _map_parameters


def _map_domain(x, old, new):
    x = numpy.asanyarray(x)
    off, scl = _map_parameters(old, new)
    return off + scl * x
