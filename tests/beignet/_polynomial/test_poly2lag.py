import beignet.polynomial
import torch.testing

from .test_polynomial import laguerre_polynomial_coefficients


def test_poly2lag():
    for index in range(7):
        torch.testing.assert_close(
            beignet.polynomial.poly2lag(
                laguerre_polynomial_coefficients[index],
            ),
            torch.tensor([0] * index + [1], dtype=torch.float64),
        )
