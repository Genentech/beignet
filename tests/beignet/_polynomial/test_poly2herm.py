import beignet.polynomial
import torch

from .test_polynomial import hermite_polynomial_coefficients


def test_poly2herm():
    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial.poly2herm(
                hermite_polynomial_coefficients[index],
            ),
            torch.tensor([0] * index + [1]),
        )
