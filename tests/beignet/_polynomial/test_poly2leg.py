import beignet.polynomial
import numpy

from tests.beignet._polynomial.test_polynomial import legendre_polynomial_Llist


def test_poly2leg():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.poly2leg(legendre_polynomial_Llist[i]), [0] * i + [1]
        )
