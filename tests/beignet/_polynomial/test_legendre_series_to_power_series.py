import beignet.polynomial
import beignet.polynomial._legendre_series_to_power_series
import torch

from tests.beignet._polynomial.test_polynomial import legendre_polynomial_coefficients


def test_legendre_series_to_power_series():
    for i in range(10):
        torch.testing.assert_close(
            beignet.polynomial.legendre_series_to_power_series([0] * i + [1]),
            legendre_polynomial_coefficients[i],
        )
