import beignet.polynomial
import beignet.polynomial._polycompanion
import numpy


def test_polycompanion():
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._polycompanion.polycompanion, []
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._polycompanion.polycompanion, [1]
    )

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(
            beignet.polynomial._polycompanion.polycompanion(coef).shape == (i, i)
        )

    numpy.testing.assert_(
        beignet.polynomial._polycompanion.polycompanion([1, 2])[0, 0] == -0.5
    )
