import beignet.polynomial
import beignet.polynomial._evaluate_1d_laguerre_series
import beignet.polynomial._lagfit
import numpy
import torch


def test_lagfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._lagfit.lagfit, [1], [1], -1
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._lagfit.lagfit, [[1]], [1], 0
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._lagfit.lagfit, [], [1], 0
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._lagfit.lagfit, [1], [[[1]]], 0
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._lagfit.lagfit, [1, 2], [1], 0
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._lagfit.lagfit, [1], [1, 2], 0
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._lagfit.lagfit, [1], [1], 0, w=[[1]]
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._lagfit.lagfit, [1], [1], 0, w=[1, 1]
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial._lagfit.lagfit,
        [1],
        [1],
        [
            -1,
        ],
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._lagfit.lagfit, [1], [1], [2, -1, 6]
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._lagfit.lagfit, [1], [1], []
    )

    x = numpy.linspace(0, 2)
    y = f(x)

    coef3 = beignet.polynomial._lagfit.lagfit(x, y, 3)
    torch.testing.assert_close(len(coef3), 4)
    numpy.testing.assert_almost_equal(
        beignet.polynomial._lagval.evaluate_1d_laguerre_series(x, coef3), y
    )
    coef3 = beignet.polynomial._lagfit.lagfit(x, y, [0, 1, 2, 3])
    torch.testing.assert_close(len(coef3), 4)
    numpy.testing.assert_almost_equal(
        beignet.polynomial._lagval.evaluate_1d_laguerre_series(x, coef3), y
    )

    coef4 = beignet.polynomial._lagfit.lagfit(x, y, 4)
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(
        beignet.polynomial._lagval.evaluate_1d_laguerre_series(x, coef4), y
    )
    coef4 = beignet.polynomial._lagfit.lagfit(x, y, [0, 1, 2, 3, 4])
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(
        beignet.polynomial._lagval.evaluate_1d_laguerre_series(x, coef4), y
    )

    coef2d = beignet.polynomial._lagfit.lagfit(x, numpy.array([y, y]).T, 3)
    numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
    coef2d = beignet.polynomial._lagfit.lagfit(x, numpy.array([y, y]).T, [0, 1, 2, 3])
    numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)

    w = numpy.zeros_like(x)
    yw = y.copy()
    w[1::2] = 1
    y[0::2] = 0
    wcoef3 = beignet.polynomial._lagfit.lagfit(x, yw, 3, w=w)
    numpy.testing.assert_almost_equal(wcoef3, coef3)
    wcoef3 = beignet.polynomial._lagfit.lagfit(x, yw, [0, 1, 2, 3], w=w)
    numpy.testing.assert_almost_equal(wcoef3, coef3)

    wcoef2d = beignet.polynomial._lagfit.lagfit(x, numpy.array([yw, yw]).T, 3, w=w)
    numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
    wcoef2d = beignet.polynomial._lagfit.lagfit(
        x, numpy.array([yw, yw]).T, [0, 1, 2, 3], w=w
    )
    numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)

    x = [1, 1j, -1, -1j]
    numpy.testing.assert_almost_equal(
        beignet.polynomial._lagfit.lagfit(x, x, 1), [1, -1]
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial._lagfit.lagfit(x, x, [0, 1]), [1, -1]
    )
