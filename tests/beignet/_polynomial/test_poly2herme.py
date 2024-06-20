import beignet.polynomial
import torch.testing

from .test_polynomial import hermite_e_polynomial_coefficients


def test_poly2herme():
    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial.poly2herme(hermite_e_polynomial_coefficients[index]),
            torch.tensor([0] * index + [1]),
        )
