import beignet.polynomial
import numpy


def test_hermcompanion():
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermcompanion, [])
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermcompanion, [1])

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(beignet.polynomial.hermcompanion(coef).shape == (i, i))

    numpy.testing.assert_(beignet.polynomial.hermcompanion([1, 2])[0, 0] == -0.25)
