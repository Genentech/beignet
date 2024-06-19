import beignet.polynomial
import beignet.polynomial._poly2herm
import numpy

from tests.beignet._polynomial.test_polynomial import hermite_polynomial_coefficients


def test_poly2herm():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._poly2herm.poly2herm(hermite_polynomial_coefficients[i]),
            [0] * i + [1],
        )
