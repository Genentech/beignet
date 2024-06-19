import beignet.polynomial
import beignet.polynomial._poly2cheb
import numpy

from tests.beignet._polynomial.test_polynomial import chebyshev_polynomial_coefficients


def test_poly2cheb():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._poly2cheb.poly2cheb(
                chebyshev_polynomial_coefficients[i]
            ),
            [0] * i + [1],
        )
