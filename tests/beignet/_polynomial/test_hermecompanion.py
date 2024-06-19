import beignet.polynomial
import beignet.polynomial._hermecompanion
import numpy


def test_hermecompanion():
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._hermecompanion.hermecompanion, []
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._hermecompanion.hermecompanion, [1]
    )

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(
            beignet.polynomial._hermecompanion.hermecompanion(coef).shape == (i, i)
        )

    numpy.testing.assert_(
        beignet.polynomial._hermecompanion.hermecompanion([1, 2])[0, 0] == -0.5
    )
