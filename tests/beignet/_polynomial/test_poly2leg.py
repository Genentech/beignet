import beignet.polynomial
import beignet.polynomial._poly2leg
import numpy

from tests.beignet._polynomial.test_polynomial import legendre_polynomial_Llist


def test_poly2leg():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._poly2leg.poly2leg(legendre_polynomial_Llist[i]),
            [0] * i + [1],
        )
