import beignet.polynomial
import numpy
import torch


def test_fit_probabilists_hermite_series():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.fit_probabilists_hermite_series,
        [1],
        [1],
        -1,
    )
    numpy.testing.assert_raises(
        TypeError,
        beignet.polynomial.fit_probabilists_hermite_series,
        [[1]],
        [1],
        0,
    )
    numpy.testing.assert_raises(
        TypeError,
        beignet.polynomial.fit_probabilists_hermite_series,
        [],
        [1],
        0,
    )
    numpy.testing.assert_raises(
        TypeError,
        beignet.polynomial.fit_probabilists_hermite_series,
        [1],
        [[[1]]],
        0,
    )
    numpy.testing.assert_raises(
        TypeError,
        beignet.polynomial.fit_probabilists_hermite_series,
        [1, 2],
        [1],
        0,
    )
    numpy.testing.assert_raises(
        TypeError,
        beignet.polynomial.fit_probabilists_hermite_series,
        [1],
        [1, 2],
        0,
    )
    numpy.testing.assert_raises(
        TypeError,
        beignet.polynomial.fit_probabilists_hermite_series,
        [1],
        [1],
        0,
        w=[[1]],
    )
    numpy.testing.assert_raises(
        TypeError,
        beignet.polynomial.fit_probabilists_hermite_series,
        [1],
        [1],
        0,
        w=[1, 1],
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.fit_probabilists_hermite_series,
        [1],
        [1],
        [
            -1,
        ],
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.fit_probabilists_hermite_series,
        [1],
        [1],
        [2, -1, 6],
    )
    numpy.testing.assert_raises(
        TypeError,
        beignet.polynomial.fit_probabilists_hermite_series,
        [1],
        [1],
        [],
    )

    coef3 = beignet.polynomial.fit_probabilists_hermite_series(
        numpy.linspace(0, 2), f(numpy.linspace(0, 2)), 3
    )
    torch.testing.assert_close(len(coef3), 4)
    torch.testing.assert_close(
        beignet.polynomial.evaluate_probabilists_hermite_series_1d(
            numpy.linspace(0, 2), coef3
        ),
        f(numpy.linspace(0, 2)),
    )
    coef3 = beignet.polynomial.fit_probabilists_hermite_series(
        numpy.linspace(0, 2), f(numpy.linspace(0, 2)), [0, 1, 2, 3]
    )
    torch.testing.assert_close(len(coef3), 4)
    torch.testing.assert_close(
        beignet.polynomial.evaluate_probabilists_hermite_series_1d(
            numpy.linspace(0, 2), coef3
        ),
        f(numpy.linspace(0, 2)),
    )

    coef4 = beignet.polynomial.fit_probabilists_hermite_series(
        numpy.linspace(0, 2), f(numpy.linspace(0, 2)), 4
    )
    torch.testing.assert_close(len(coef4), 5)
    torch.testing.assert_close(
        beignet.polynomial.evaluate_probabilists_hermite_series_1d(
            numpy.linspace(0, 2), coef4
        ),
        f(numpy.linspace(0, 2)),
    )
    coef4 = beignet.polynomial.fit_probabilists_hermite_series(
        numpy.linspace(0, 2), f(numpy.linspace(0, 2)), [0, 1, 2, 3, 4]
    )
    torch.testing.assert_close(len(coef4), 5)
    torch.testing.assert_close(
        beignet.polynomial.evaluate_probabilists_hermite_series_1d(
            numpy.linspace(0, 2), coef4
        ),
        f(numpy.linspace(0, 2)),
    )

    coef4 = beignet.polynomial.fit_probabilists_hermite_series(
        numpy.linspace(0, 2), f(numpy.linspace(0, 2)), [2, 3, 4, 1, 0]
    )
    torch.testing.assert_close(len(coef4), 5)
    torch.testing.assert_close(
        beignet.polynomial.evaluate_probabilists_hermite_series_1d(
            numpy.linspace(0, 2), coef4
        ),
        f(numpy.linspace(0, 2)),
    )

    coef2d = beignet.polynomial.fit_probabilists_hermite_series(
        numpy.linspace(0, 2),
        numpy.array([(f(numpy.linspace(0, 2))), (f(numpy.linspace(0, 2)))]).T,
        3,
    )
    torch.testing.assert_close(coef2d, numpy.array([coef3, coef3]).T)
    coef2d = beignet.polynomial.fit_probabilists_hermite_series(
        numpy.linspace(0, 2),
        numpy.array([(f(numpy.linspace(0, 2))), (f(numpy.linspace(0, 2)))]).T,
        [0, 1, 2, 3],
    )
    torch.testing.assert_close(coef2d, numpy.array([coef3, coef3]).T)

    w = numpy.zeros_like(numpy.linspace(0, 2))
    yw = f(numpy.linspace(0, 2)).copy()
    w[1::2] = 1
    f(numpy.linspace(0, 2))[0::2] = 0
    wcoef3 = beignet.polynomial.fit_probabilists_hermite_series(
        numpy.linspace(0, 2), yw, 3, w=w
    )
    torch.testing.assert_close(wcoef3, coef3)
    wcoef3 = beignet.polynomial.fit_probabilists_hermite_series(
        numpy.linspace(0, 2), yw, [0, 1, 2, 3], w=w
    )
    torch.testing.assert_close(wcoef3, coef3)

    wcoef2d = beignet.polynomial.fit_probabilists_hermite_series(
        numpy.linspace(0, 2), numpy.array([yw, yw]).T, 3, w=w
    )
    torch.testing.assert_close(wcoef2d, numpy.array([coef3, coef3]).T)
    wcoef2d = beignet.polynomial.fit_probabilists_hermite_series(
        numpy.linspace(0, 2), numpy.array([yw, yw]).T, [0, 1, 2, 3], w=w
    )
    torch.testing.assert_close(wcoef2d, numpy.array([coef3, coef3]).T)

    x = [1, 1j, -1, -1j]
    torch.testing.assert_close(
        beignet.polynomial.fit_probabilists_hermite_series(x, x, 1), [0, 1]
    )
    torch.testing.assert_close(
        beignet.polynomial.fit_probabilists_hermite_series(x, x, [0, 1]),
        [0, 1],
    )

    torch.testing.assert_close(
        beignet.polynomial.evaluate_probabilists_hermite_series_1d(
            numpy.linspace(-1, 1),
            beignet.polynomial.fit_probabilists_hermite_series(
                numpy.linspace(-1, 1), g(numpy.linspace(-1, 1)), 4
            ),
        ),
        g(numpy.linspace(-1, 1)),
    )
    torch.testing.assert_close(
        beignet.polynomial.evaluate_probabilists_hermite_series_1d(
            numpy.linspace(-1, 1),
            beignet.polynomial.fit_probabilists_hermite_series(
                numpy.linspace(-1, 1), g(numpy.linspace(-1, 1)), [0, 2, 4]
            ),
        ),
        g(numpy.linspace(-1, 1)),
    )
    torch.testing.assert_close(
        beignet.polynomial.fit_probabilists_hermite_series(
            numpy.linspace(-1, 1), g(numpy.linspace(-1, 1)), 4
        ),
        beignet.polynomial.fit_probabilists_hermite_series(
            numpy.linspace(-1, 1), g(numpy.linspace(-1, 1)), [0, 2, 4]
        ),
    )
