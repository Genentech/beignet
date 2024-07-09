import beignet
import torch


def test_multiply_chebyshev_polynomial_by_x():
    torch.testing.assert_close(
        beignet.trim_chebyshev_polynomial_coefficients(
            beignet.multiply_chebyshev_polynomial_by_x(
                torch.tensor([0.0]),
            ),
            tol=0.000001,
        ),
        torch.tensor([0.0]),
    )

    torch.testing.assert_close(
        beignet.trim_chebyshev_polynomial_coefficients(
            beignet.multiply_chebyshev_polynomial_by_x(
                torch.tensor([1.0]),
            ),
            tol=0.000001,
        ),
        torch.tensor([0.0, 1.0]),
    )

    for index in range(1, 5):
        torch.testing.assert_close(
            beignet.trim_chebyshev_polynomial_coefficients(
                beignet.multiply_chebyshev_polynomial_by_x(
                    torch.tensor([0.0] * index + [1.0]),
                ),
                tol=0.000001,
            ),
            torch.tensor([0.0] * (index - 1) + [0.5, 0, 0.5]),
        )
