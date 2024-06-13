import numpy


def chebpts2(npts):
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 2:
        raise ValueError("npts must be >= 2")

    x = numpy.linspace(-numpy.pi, 0, _npts)
    return numpy.cos(x)
