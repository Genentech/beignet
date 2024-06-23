import beignet.polynomial
import numpy
import torch

from tests.beignet._polynomial.test_polynomial import chebyshev_polynomial_coefficients


def test_evaluate_chebyshev_series_1d():
    torch.testing.assert_close(
        beignet.polynomial.evaluate_chebyshev_series_1d([], [1]).size, 0
    )

    x = numpy.linspace(-1, 1)
    y = [
        beignet.polynomial.evaluate_power_series_1d(x, c)
        for c in chebyshev_polynomial_coefficients
    ]
    for i in range(10):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial.evaluate_chebyshev_series_1d(x, [0] * i + [1])
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(
            beignet.polynomial.evaluate_chebyshev_series_1d(x, [1]).shape, dims
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_chebyshev_series_1d(x, [1, 0]).shape,
            dims,
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_chebyshev_series_1d(x, [1, 0, 0]).shape,
            dims,
        )
