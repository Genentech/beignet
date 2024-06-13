import numpy

from .__zseries_mul import _zseries_mul


def _zseries_int(zs):
    n = 1 + len(zs) // 2
    ns = numpy.array([-1, 0, 1], dtype=zs.dtype)
    zs = _zseries_mul(zs, ns)
    div = numpy.arange(-n, n + 1) * 2
    zs[:n] /= div[:n]
    zs[n + 1 :] /= div[n + 1 :]
    zs[n] = 0
    return zs
