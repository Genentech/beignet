import beignet
import torch


def test_multiply_legendre_polynomial_by_x():
    torch.testing.assert_close(
        beignet.trim_legendre_polynomial_coefficients(
            beignet.multiply_legendre_polynomial_by_x(
                torch.tensor([0.0]),
            ),
            tol=0.000001,
        ),
        torch.tensor([0.0]),
    )

    torch.testing.assert_close(
        beignet.trim_legendre_polynomial_coefficients(
            beignet.multiply_legendre_polynomial_by_x(
                torch.tensor([1.0]),
            ),
            tol=0.000001,
        ),
        torch.tensor([0.0, 1.0]),
    )

    for i in range(1, 5):
        torch.testing.assert_close(
            beignet.trim_legendre_polynomial_coefficients(
                beignet.multiply_legendre_polynomial_by_x(
                    torch.tensor([0.0] * i + [1.0]),
                ),
                tol=0.000001,
            ),
            torch.tensor([0] * (i - 1) + [i / (2 * i + 1), 0, (i + 1) / (2 * i + 1)]),
        )
