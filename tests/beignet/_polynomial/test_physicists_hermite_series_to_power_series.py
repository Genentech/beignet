import beignet.polynomial
import beignet.polynomial._physicists_hermite_series_to_power_series
import torch

from tests.beignet._polynomial.test_polynomial import hermite_polynomial_coefficients


def test_physicists_hermite_series_to_power_series():
    for i in range(10):
        torch.testing.assert_close(
            beignet.polynomial._herm2poly.physicists_hermite_series_to_power_series(
                [0] * i + [1]
            ),
            hermite_polynomial_coefficients[i],
        )
