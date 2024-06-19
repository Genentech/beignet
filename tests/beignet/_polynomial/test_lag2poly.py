import beignet.polynomial
import beignet.polynomial._lag2poly
import numpy

from tests.beignet._polynomial.test_polynomial import laguerre_polynomial_coefficients


def test_lag2poly():
    for i in range(7):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lag2poly.lag2poly([0] * i + [1]),
            laguerre_polynomial_coefficients[i],
        )
