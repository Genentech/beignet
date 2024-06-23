import beignet.polynomial
import torch

from .test_polynomial import laguerre_polynomial_coefficients


def test_laguerre_series_to_power_series():
    for i in range(7):
        torch.testing.assert_close(
            beignet.polynomial.laguerre_series_to_power_series([0] * i + [1]),
            laguerre_polynomial_coefficients[i],
        )
