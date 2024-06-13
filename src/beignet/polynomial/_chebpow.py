import numpy

from .__cseries_to_zseries import _cseries_to_zseries
from .__zseries_to_cseries import _zseries_to_cseries
from ._as_series import as_series


def chebpow(c, pow, maxpower=16):
    [c] = as_series([c])
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    elif maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    elif power == 0:
        return numpy.array([1], dtype=c.dtype)
    elif power == 1:
        return c
    else:
        zs = _cseries_to_zseries(c)
        prd = zs
        for _ in range(2, power + 1):
            prd = numpy.convolve(prd, zs)
        return _zseries_to_cseries(prd)
