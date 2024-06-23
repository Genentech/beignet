import beignet.polynomial
import numpy
import torch


def test_chebinterpolate():
    def func(x):
        return x * (x - 1) * (x - 2)

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.chebinterpolate, func, -1
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.chebinterpolate, func, 10.0
    )

    for deg in range(1, 5):
        numpy.testing.assert_(
            beignet.polynomial.chebinterpolate(func, deg).shape == (deg + 1,)
        )

    def powx(x, p):
        return x**p

    x = numpy.linspace(-1, 1, 10)
    for deg in range(0, 10):
        for p in range(0, deg + 1):
            c = beignet.polynomial.chebinterpolate(powx, deg, (p,))
            torch.testing.assert_close(
                beignet.polynomial.evaluate_chebyshev_series_1d(x, c),
                powx(x, p),
                decimal=12,
            )
