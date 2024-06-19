import beignet.polynomial
import numpy

from tests.beignet._polynomial.test_polynomial import legendre_polynomial_Llist


def test_legval():
    numpy.testing.assert_equal(beignet.polynomial.legval([], [1]).size, 0)

    x = numpy.linspace(-1, 1)
    y = [beignet.polynomial.polyval(x, c) for c in legendre_polynomial_Llist]
    for i in range(10):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial.legval(x, [0] * i + [1])
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        numpy.testing.assert_equal(beignet.polynomial.legval(x, [1]).shape, dims)
        numpy.testing.assert_equal(beignet.polynomial.legval(x, [1, 0]).shape, dims)
        numpy.testing.assert_equal(beignet.polynomial.legval(x, [1, 0, 0]).shape, dims)
