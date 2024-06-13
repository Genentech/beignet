import numpy


def polyvalfromroots(x, r, tensor=True):
    r = numpy.array(r, ndmin=1, copy=False)

    if r.dtype.char in "?bBhHiIlLqQpP":
        r = r.astype(numpy.double)

    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)

    if isinstance(x, numpy.ndarray):
        if tensor:
            r = r.reshape(r.shape + (1,) * x.ndim)
        elif x.ndim >= r.ndim:
            raise ValueError("x.ndim must be < r.ndim when tensor == False")

    return numpy.prod(x - r, axis=0)
