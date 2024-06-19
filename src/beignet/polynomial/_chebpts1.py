import numpy


def chebpts1(npts):
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 1:
        raise ValueError("npts must be >= 1")

    x = 0.5 * numpy.pi / _npts * numpy.arange(-_npts + 1, _npts + 1, 2)
    return numpy.sin(x)
