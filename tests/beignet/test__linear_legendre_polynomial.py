import beignet
import torch


def test_linear_legendre_polynomial():
    torch.testing.assert_close(
        beignet.linear_legendre_polynomial(3.0, 4.0),
        torch.tensor([3.0, 4.0]),
    )

    torch.testing.assert_close(
        beignet.trim_legendre_polynomial_coefficients(
            beignet.linear_legendre_polynomial(3.0, 0.0),
            tol=0.000001,
        ),
        torch.tensor([3.0]),
    )
