import beignet.polynomial
import numpy

from tests.beignet._polynomial.test_polynomial import hermite_polynomial_Hlist


def test_herm2poly():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.herm2poly([0] * i + [1]),
            hermite_polynomial_Hlist[i],
        )
