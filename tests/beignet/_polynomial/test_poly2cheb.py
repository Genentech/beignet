import beignet.polynomial
import numpy

from tests.beignet._polynomial.test_polynomial import chebyshev_polynomial_Tlist


def test_poly2cheb():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.poly2cheb(chebyshev_polynomial_Tlist[i]),
            [0] * i + [1],
        )
