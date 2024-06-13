import numpy

from ._as_series import as_series


def _pow(mul_f, c, pow, maxpower):
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
        prd = c
        for _ in range(2, power + 1):
            prd = mul_f(prd, c)
        return prd
