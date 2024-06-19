import beignet.polynomial
import beignet.polynomial._poly2lag
import numpy

from tests.beignet._polynomial.test_polynomial import laguerre_polynomial_Llist


def test_poly2lag():
    for i in range(7):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._poly2lag.poly2lag(laguerre_polynomial_Llist[i]),
            [0] * i + [1],
        )
