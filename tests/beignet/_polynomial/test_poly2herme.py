import beignet.polynomial
import beignet.polynomial._poly2herme
import numpy

from tests.beignet._polynomial.test_polynomial import hermite_e_polynomial_coefficients


def test_poly2herme():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._poly2herme.poly2herme(
                hermite_e_polynomial_coefficients[i]
            ),
            [0] * i + [1],
        )
