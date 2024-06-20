import beignet.polynomial
import beignet.polynomial._chebinterpolate
import beignet.polynomial._evaluate_1d_chebyshev_series
import numpy


def test_chebinterpolate():
    def func(x):
        return x * (x - 1) * (x - 2)

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._chebinterpolate.chebinterpolate, func, -1
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._chebinterpolate.chebinterpolate, func, 10.0
    )

    for deg in range(1, 5):
        numpy.testing.assert_(
            beignet.polynomial._chebinterpolate.chebinterpolate(func, deg).shape
            == (deg + 1,)
        )

    def powx(x, p):
        return x**p

    x = numpy.linspace(-1, 1, 10)
    for deg in range(0, 10):
        for p in range(0, deg + 1):
            c = beignet.polynomial._chebinterpolate.chebinterpolate(powx, deg, (p,))
            numpy.testing.assert_almost_equal(
                beignet.polynomial._chebval.evaluate_1d_chebyshev_series(x, c),
                powx(x, p),
                decimal=12,
            )
