import beignet.polynomial
import beignet.polynomial._herme2poly
import numpy

from tests.beignet._polynomial.test_polynomial import hermite_e_polynomial_coefficients


def test_herme2poly():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._herme2poly.herme2poly([0] * i + [1]),
            hermite_e_polynomial_coefficients[i],
        )
