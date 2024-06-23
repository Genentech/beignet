import beignet.polynomial
import beignet.polynomial._evaluate_1d_power_series
import beignet.polynomial._evaluate_chebyshev_series_1d
import numpy
import torch

from tests.beignet._polynomial.test_polynomial import chebyshev_polynomial_coefficients


def test_evaluate_1d_chebyshev_series():
    torch.testing.assert_close(
        beignet.polynomial._chebval.evaluate_chebyshev_series_1d([], [1]).size, 0
    )

    x = numpy.linspace(-1, 1)
    y = [
        beignet.polynomial._polyval.evaluate_1d_power_series(x, c)
        for c in chebyshev_polynomial_coefficients
    ]
    for i in range(10):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial._chebval.evaluate_chebyshev_series_1d(x, [0] * i + [1])
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(
            beignet.polynomial._chebval.evaluate_chebyshev_series_1d(x, [1]).shape, dims
        )
        torch.testing.assert_close(
            beignet.polynomial._chebval.evaluate_chebyshev_series_1d(x, [1, 0]).shape,
            dims,
        )
        torch.testing.assert_close(
            beignet.polynomial._chebval.evaluate_chebyshev_series_1d(
                x, [1, 0, 0]
            ).shape,
            dims,
        )
