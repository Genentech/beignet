import beignet.polynomial
import numpy


def test_polycompanion():
    numpy.testing.assert_raises(ValueError, beignet.polynomial.polycompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.polycompanion, [1])

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(beignet.polynomial.polycompanion(coef).shape == (i, i))

    numpy.testing.assert_(beignet.polynomial.polycompanion([1, 2])[0, 0] == -0.5)
