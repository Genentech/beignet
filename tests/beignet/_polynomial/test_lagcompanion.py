import beignet.polynomial
import beignet.polynomial._lagcompanion
import numpy


def test_lagcompanion():
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._lagcompanion.lagcompanion, []
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._lagcompanion.lagcompanion, [1]
    )

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(
            beignet.polynomial._lagcompanion.lagcompanion(coef).shape == (i, i)
        )

    numpy.testing.assert_(
        beignet.polynomial._lagcompanion.lagcompanion([1, 2])[0, 0] == 1.5
    )
