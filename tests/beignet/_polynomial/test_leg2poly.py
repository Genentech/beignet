import beignet.polynomial
import beignet.polynomial._leg2poly
import numpy

from tests.beignet._polynomial.test_polynomial import legendre_polynomial_Llist


def test_leg2poly():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._leg2poly.leg2poly([0] * i + [1]),
            legendre_polynomial_Llist[i],
        )
