import beignet.polynomial
import beignet.polynomial._legendre_series_companion
import numpy


def test_legcompanion():
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._legcompanion.legendre_series_companion, []
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._legcompanion.legendre_series_companion, [1]
    )

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(
            beignet.polynomial._legcompanion.legendre_series_companion(coef).shape
            == (i, i)
        )

    numpy.testing.assert_(
        beignet.polynomial._legcompanion.legendre_series_companion([1, 2])[0, 0] == -0.5
    )
