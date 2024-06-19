import beignet.polynomial
import numpy


def test_chebpts2():
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebpts2, 1.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebpts2, 1)
    numpy.testing.assert_almost_equal(beignet.polynomial.chebpts2(2), [-1, 1])
    numpy.testing.assert_almost_equal(beignet.polynomial.chebpts2(3), [-1, 0, 1])
    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebpts2(4), [-1, -0.5, 0.5, 1]
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebpts2(5), [-1.0, -0.707106781187, 0, 0.707106781187, 1.0]
    )
