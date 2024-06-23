import numpy


def evaluate_legendre_series_1d(x, c, tensor=True):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)
    if isinstance(x, numpy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * x * (2 * nd - 1)) / nd
    return c0 + c1 * x
