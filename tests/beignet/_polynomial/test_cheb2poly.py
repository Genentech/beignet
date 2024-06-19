import beignet.polynomial
import numpy

from tests.beignet._polynomial.test_polynomial import chebyshev_polynomial_Tlist


def test_cheb2poly():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.cheb2poly([0] * i + [1]),
            chebyshev_polynomial_Tlist[i],
        )
