import beignet.polynomial
import torch

from tests.beignet._polynomial.test_polynomial import hermite_e_polynomial_coefficients


def test_probabilists_hermite_series_to_power_series():
    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial.probabilists_hermite_series_to_power_series(
                torch.tensor([0] * index + [1])
            ),
            hermite_e_polynomial_coefficients[index],
        )
