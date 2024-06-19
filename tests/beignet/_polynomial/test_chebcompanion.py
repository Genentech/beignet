import beignet.polynomial
import numpy


def test_chebcompanion():
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebcompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebcompanion, [1])

    for i in range(1, 5):
        numpy.testing.assert_(
            beignet.polynomial.chebcompanion([0] * i + [1]).shape == (i, i)
        )

    numpy.testing.assert_(beignet.polynomial.chebcompanion([1, 2])[0, 0] == -0.5)
