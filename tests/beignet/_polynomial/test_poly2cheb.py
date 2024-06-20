import beignet.polynomial
import torch.testing

from .test_polynomial import chebyshev_polynomial_coefficients


def test_poly2cheb():
    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial.power_series_to_chebyshev_series(
                chebyshev_polynomial_coefficients[index],
            ),
            torch.tensor([0] * index + [1], dtype=torch.float64),
        )
