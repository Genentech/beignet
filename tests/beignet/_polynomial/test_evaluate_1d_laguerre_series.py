import beignet.polynomial
import beignet.polynomial._evaluate_laguerre_series_1d
import beignet.polynomial._evaluate_power_series_1d
import numpy
import torch

from tests.beignet._polynomial.test_polynomial import laguerre_polynomial_coefficients


def test_evaluate_1d_laguerre_series():
    x = numpy.random.random((3, 5)) * 2 - 1

    torch.testing.assert_close(
        beignet.polynomial._lagval.evaluate_laguerre_series_1d([], [1]).size, 0
    )

    x = numpy.linspace(-1, 1)
    y = [
        beignet.polynomial._polyval.evaluate_power_series_1d(x, c)
        for c in laguerre_polynomial_coefficients
    ]
    for i in range(7):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial._lagval.evaluate_laguerre_series_1d(x, [0] * i + [1])
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(
            beignet.polynomial._lagval.evaluate_laguerre_series_1d(x, [1]).shape, dims
        )
        torch.testing.assert_close(
            beignet.polynomial._lagval.evaluate_laguerre_series_1d(x, [1, 0]).shape,
            dims,
        )
        torch.testing.assert_close(
            beignet.polynomial._lagval.evaluate_laguerre_series_1d(x, [1, 0, 0]).shape,
            dims,
        )
