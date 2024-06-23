import beignet.polynomial
import torch

from .test_polynomial import chebyshev_polynomial_coefficients


def test_chebyshev_series_to_power_series():
    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial.chebyshev_series_to_power_series(
                torch.tensor([0] * index + [1]),
            ),
            chebyshev_polynomial_coefficients[index],
        )
