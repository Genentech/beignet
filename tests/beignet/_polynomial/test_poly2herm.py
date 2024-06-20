import beignet.polynomial
import torch

from .test_polynomial import hermite_polynomial_coefficients


def test_poly2herm():
    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial.power_series_to_physicists_hermite_series(
                hermite_polynomial_coefficients[index],
            ),
            torch.tensor([0] * index + [1], dtype=torch.float64),
        )
