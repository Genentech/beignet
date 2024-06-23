import beignet.polynomial
import torch.testing

from tests.beignet._polynomial.test_polynomial import chebyshev_polynomial_coefficients


def test_power_series_to_chebyshev_series():
    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial.power_series_to_chebyshev_series(
                chebyshev_polynomial_coefficients[index],
            ),
            torch.tensor([0] * index + [1], dtype=torch.float64),
        )
