import beignet.polynomial
import beignet.polynomial._evaluate_legendre_series_1d
import beignet.polynomial._evaluate_power_series_1d
import numpy
import torch

from tests.beignet._polynomial.test_polynomial import legendre_polynomial_coefficients


def test_evaluate_1d_legendre_series():
    torch.testing.assert_close(
        beignet.polynomial.evaluate_legendre_series_1d([], [1]).size,
        0,
    )

    x = numpy.linspace(-1, 1)

    y = [
        beignet.polynomial.evaluate_power_series_1d(x, c)
        for c in legendre_polynomial_coefficients
    ]

    for i in range(10):
        torch.testing.assert_close(
            beignet.polynomial.evaluate_legendre_series_1d(x, [0] * i + [1]),
            y[i],
        )

    for i in range(3):
        dims = [2] * i

        x = numpy.zeros(dims)

        torch.testing.assert_close(
            beignet.polynomial.evaluate_legendre_series_1d(x, [1]).shape,
            dims,
        )

        torch.testing.assert_close(
            beignet.polynomial.evaluate_legendre_series_1d(x, [1, 0]).shape,
            dims,
        )

        torch.testing.assert_close(
            beignet.polynomial.evaluate_legendre_series_1d(x, [1, 0, 0]).shape,
            dims,
        )
