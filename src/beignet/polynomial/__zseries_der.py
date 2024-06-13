import numpy

from .__zseries_div import _zseries_div


def _zseries_der(zs):
    n = len(zs) // 2
    ns = numpy.array([-1, 0, 1], dtype=zs.dtype)
    zs *= numpy.arange(-n, n + 1) * 2
    d, r = _zseries_div(zs, ns)
    return d
