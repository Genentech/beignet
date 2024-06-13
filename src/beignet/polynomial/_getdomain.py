import numpy

from ._as_series import as_series


def getdomain(x):
    [x] = as_series([x], trim=False)
    if x.dtype.char in numpy.typecodes["Complex"]:
        rmin, rmax = x.real.min(), x.real.max()
        imin, imax = x.imag.min(), x.imag.max()
        return numpy.array((complex(rmin, imin), complex(rmax, imax)))
    else:
        return numpy.array((x.min(), x.max()))
