import beignet.polynomial
import torch.testing

from tests.beignet._polynomial.test_polynomial import laguerre_polynomial_coefficients


def test_power_series_to_laguerre_series():
    for index in range(7):
        torch.testing.assert_close(
            beignet.polynomial.power_series_to_laguerre_series(
                laguerre_polynomial_coefficients[index],
            ),
            torch.tensor([0] * index + [1], dtype=torch.float32),
        )
