import numpy


def _normed_hermite_e_n(x, n):
    if n == 0:
        return numpy.full(x.shape, 1 / numpy.sqrt(numpy.sqrt(2 * numpy.pi)))

    c0 = 0.0
    c1 = 1.0 / numpy.sqrt(numpy.sqrt(2 * numpy.pi))
    nd = float(n)
    for _ in range(n - 1):
        tmp = c0
        c0 = -c1 * numpy.sqrt((nd - 1.0) / nd)
        c1 = tmp + c1 * x * numpy.sqrt(1.0 / nd)
        nd = nd - 1.0
    return c0 + c1 * x
