import beignet.polynomial
import numpy
import torch

laguerre_polynomial_coefficients = [
    (torch.tensor([1]) / 1),
    (torch.tensor([1, -1]) / 1),
    (torch.tensor([2, -4, 1]) / 2),
    (torch.tensor([6, -18, 9, -1]) / 6),
    (torch.tensor([24, -96, 72, -16, 1]) / 24),
    (torch.tensor([120, -600, 600, -200, 25, -1]) / 120),
    (torch.tensor([720, -4320, 5400, -2400, 450, -36, 1]) / 720),
]


def test_evaluate_laguerre_series_1d():
    x = numpy.random.random((3, 5)) * 2 - 1

    torch.testing.assert_close(
        beignet.polynomial.evaluate_laguerre_series_1d([], [1]).size, 0
    )

    x = numpy.linspace(-1, 1)
    y = [
        beignet.polynomial.evaluate_power_series_1d(x, c)
        for c in laguerre_polynomial_coefficients
    ]
    for i in range(7):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial.evaluate_laguerre_series_1d(x, [0] * i + [1])
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(
            beignet.polynomial.evaluate_laguerre_series_1d(x, [1]).shape, dims
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_laguerre_series_1d(x, [1, 0]).shape,
            dims,
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_laguerre_series_1d(x, [1, 0, 0]).shape,
            dims,
        )
