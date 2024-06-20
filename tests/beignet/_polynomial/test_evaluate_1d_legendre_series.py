import beignet.polynomial
import beignet.polynomial._evaluate_1d_legendre_series
import beignet.polynomial._evaluate_1d_power_series
import numpy
import torch

from tests.beignet._polynomial.test_polynomial import legendre_polynomial_coefficients


def test_evaluate_1d_legendre_series():
    torch.testing.assert_close(
        beignet.polynomial.evaluate_1d_legendre_series([], [1]).size,
        0,
    )

    x = numpy.linspace(-1, 1)

    y = [
        beignet.polynomial.evaluate_1d_power_series(x, c)
        for c in legendre_polynomial_coefficients
    ]

    for i in range(10):
        torch.testing.assert_close(
            beignet.polynomial.evaluate_1d_legendre_series(x, [0] * i + [1]),
            y[i],
        )

    for i in range(3):
        dims = [2] * i

        x = numpy.zeros(dims)

        torch.testing.assert_close(
            beignet.polynomial.evaluate_1d_legendre_series(x, [1]).shape,
            dims,
        )

        torch.testing.assert_close(
            beignet.polynomial.evaluate_1d_legendre_series(x, [1, 0]).shape,
            dims,
        )

        torch.testing.assert_close(
            beignet.polynomial.evaluate_1d_legendre_series(x, [1, 0, 0]).shape,
            dims,
        )
