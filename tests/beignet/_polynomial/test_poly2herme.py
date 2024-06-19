import beignet.polynomial
import numpy

from tests.beignet._polynomial.test_polynomial import hermite_e_polynomial_Helist


def test_poly2herme():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.poly2herme(hermite_e_polynomial_Helist[i]),
            [0] * i + [1],
        )
