import beignet.polynomial
import beignet.polynomial._poly2leg
import torch.testing

from tests.beignet._polynomial.test_polynomial import legendre_polynomial_coefficients


def test_poly2leg():
    for i in range(10):
        torch.testing.assert_close(
            beignet.polynomial.poly2leg(
                legendre_polynomial_coefficients[i],
            ),
            torch.tensor([0] * i + [1]),
        )
