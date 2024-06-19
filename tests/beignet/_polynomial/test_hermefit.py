import beignet.polynomial
import numpy
import torch


def test_hermefit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def f2(x):
        return x**4 + x**2 + 1

    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermefit, [1], [1], -1)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermefit, [[1]], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermefit, [], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermefit, [1], [[[1]]], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermefit, [1, 2], [1], 0)
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermefit, [1], [1, 2], 0)
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.hermefit, [1], [1], 0, w=[[1]]
    )
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.hermefit, [1], [1], 0, w=[1, 1]
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.hermefit,
        [1],
        [1],
        [
            -1,
        ],
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.hermefit, [1], [1], [2, -1, 6]
    )
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermefit, [1], [1], [])

    x = numpy.linspace(0, 2)
    y = f(x)

    coef3 = beignet.polynomial.hermefit(x, y, 3)
    torch.testing.assert_close(len(coef3), 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(x, coef3), y)
    coef3 = beignet.polynomial.hermefit(x, y, [0, 1, 2, 3])
    torch.testing.assert_close(len(coef3), 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(x, coef3), y)

    coef4 = beignet.polynomial.hermefit(x, y, 4)
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(x, coef4), y)
    coef4 = beignet.polynomial.hermefit(x, y, [0, 1, 2, 3, 4])
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(x, coef4), y)

    coef4 = beignet.polynomial.hermefit(x, y, [2, 3, 4, 1, 0])
    torch.testing.assert_close(len(coef4), 5)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(x, coef4), y)

    coef2d = beignet.polynomial.hermefit(x, numpy.array([y, y]).T, 3)
    numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)
    coef2d = beignet.polynomial.hermefit(x, numpy.array([y, y]).T, [0, 1, 2, 3])
    numpy.testing.assert_almost_equal(coef2d, numpy.array([coef3, coef3]).T)

    w = numpy.zeros_like(x)
    yw = y.copy()
    w[1::2] = 1
    y[0::2] = 0
    wcoef3 = beignet.polynomial.hermefit(x, yw, 3, w=w)
    numpy.testing.assert_almost_equal(wcoef3, coef3)
    wcoef3 = beignet.polynomial.hermefit(x, yw, [0, 1, 2, 3], w=w)
    numpy.testing.assert_almost_equal(wcoef3, coef3)

    wcoef2d = beignet.polynomial.hermefit(x, numpy.array([yw, yw]).T, 3, w=w)
    numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)
    wcoef2d = beignet.polynomial.hermefit(x, numpy.array([yw, yw]).T, [0, 1, 2, 3], w=w)
    numpy.testing.assert_almost_equal(wcoef2d, numpy.array([coef3, coef3]).T)

    x = [1, 1j, -1, -1j]
    numpy.testing.assert_almost_equal(beignet.polynomial.hermefit(x, x, 1), [0, 1])
    numpy.testing.assert_almost_equal(beignet.polynomial.hermefit(x, x, [0, 1]), [0, 1])

    x = numpy.linspace(-1, 1)
    y = f2(x)
    coef1 = beignet.polynomial.hermefit(x, y, 4)
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(x, coef1), y)
    coef2 = beignet.polynomial.hermefit(x, y, [0, 2, 4])
    numpy.testing.assert_almost_equal(beignet.polynomial.hermeval(x, coef2), y)
    numpy.testing.assert_almost_equal(coef1, coef2)
