import beignet.polynomial
import torch.testing

from .test_polynomial import hermite_e_polynomial_coefficients


def test_poly2herme():
    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial.power_series_to_probabilists_hermite_series(
                hermite_e_polynomial_coefficients[index],
            ),
            torch.tensor([0] * index + [1], dtype=torch.float64),
        )
