import beignet.polynomial
import beignet.polynomial._legcompanion
import numpy


def test_legcompanion():
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._legcompanion.legcompanion, []
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._legcompanion.legcompanion, [1]
    )

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(
            beignet.polynomial._legcompanion.legcompanion(coef).shape == (i, i)
        )

    numpy.testing.assert_(
        beignet.polynomial._legcompanion.legcompanion([1, 2])[0, 0] == -0.5
    )
