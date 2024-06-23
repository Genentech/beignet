import beignet.polynomial
import numpy


def test_legendre_series_companion():
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.legendre_series_companion, []
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.legendre_series_companion, [1]
    )

    for i in range(1, 5):
        coef = [0] * i + [1]
        numpy.testing.assert_(
            beignet.polynomial.legendre_series_companion(coef).shape == (i, i)
        )

    numpy.testing.assert_(
        beignet.polynomial.legendre_series_companion([1, 2])[0, 0] == -0.5
    )
