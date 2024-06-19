import beignet.polynomial
import numpy


def test_lagcompanion():
    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagcompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagcompanion, [1])

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(beignet.polynomial.lagcompanion(coef).shape == (i, i))

    numpy.testing.assert_(beignet.polynomial.lagcompanion([1, 2])[0, 0] == 1.5)
