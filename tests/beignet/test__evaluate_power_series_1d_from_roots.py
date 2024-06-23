import beignet.polynomial
import numpy
import torch


def test_evaluate_power_series_1d_from_roots():
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.evaluate_power_series_1d_from_roots,
        [1],
        [1],
        tensor=False,
    )

    torch.testing.assert_close(
        beignet.polynomial.evaluate_power_series_1d_from_roots([], [1]).size, 0
    )
    assert beignet.polynomial.evaluate_power_series_1d_from_roots([], [1]).shape == (0,)

    torch.testing.assert_close(
        beignet.polynomial.evaluate_power_series_1d_from_roots([], [[1] * 5]).size, 0
    )
    assert beignet.polynomial.evaluate_power_series_1d_from_roots(
        [], [[1] * 5]
    ).shape == (5, 0)

    torch.testing.assert_close(
        beignet.polynomial.evaluate_power_series_1d_from_roots(1, 1), 0
    )
    assert beignet.polynomial.evaluate_power_series_1d_from_roots(
        1, numpy.ones((3, 3))
    ).shape == (3,)

    x = numpy.linspace(-1, 1)
    y = [x**i for i in range(5)]
    for i in range(1, 5):
        torch.testing.assert_close(
            beignet.polynomial.evaluate_power_series_1d_from_roots(x, [0] * i), y[i]
        )
    tgt = x * (x - 1) * (x + 1)
    torch.testing.assert_close(
        beignet.polynomial.evaluate_power_series_1d_from_roots(x, [-1, 0, 1]), tgt
    )

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(
            beignet.polynomial.evaluate_power_series_1d_from_roots(x, [1]).shape, dims
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_power_series_1d_from_roots(x, [1, 0]).shape,
            dims,
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_power_series_1d_from_roots(x, [1, 0, 0]).shape,
            dims,
        )

    ptest = [15, 2, -16, -2, 1]
    x = numpy.linspace(-1, 1)
    torch.testing.assert_close(
        beignet.polynomial.evaluate_power_series_1d(x, ptest),
        beignet.polynomial.evaluate_power_series_1d_from_roots(
            x, beignet.polynomial.power_series_roots(ptest)
        ),
    )

    rshape = (3, 5)
    x = torch.arange(-3, 2)
    r = numpy.random.randint(-5, 5, size=rshape)
    tgt = numpy.empty(r.shape[1:])
    for ii in range(tgt.size):
        tgt[ii] = beignet.polynomial.evaluate_power_series_1d_from_roots(
            x[ii], r[:, ii]
        )
    torch.testing.assert_close(
        beignet.polynomial.evaluate_power_series_1d_from_roots(x, r, tensor=False), tgt
    )

    x = numpy.vstack([x, 2 * x])
    tgt = numpy.empty(r.shape[1:] + x.shape)
    for ii in range(r.shape[1]):
        for jj in range(x.shape[0]):
            tgt[ii, jj, :] = beignet.polynomial.evaluate_power_series_1d_from_roots(
                x[jj], r[:, ii]
            )

    torch.testing.assert_close(
        beignet.polynomial.evaluate_power_series_1d_from_roots(x, r, tensor=True), tgt
    )
