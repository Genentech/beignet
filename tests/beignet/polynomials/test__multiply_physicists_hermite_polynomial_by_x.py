import torch

import beignet.polynomials


def test_multiply_physicists_hermite_polynomial_by_x(float64):
    torch.testing.assert_close(
        beignet.polynomials.trim_physicists_hermite_polynomial_coefficients(
            beignet.polynomials.multiply_physicists_hermite_polynomial_by_x(
                torch.tensor([0.0]),
            ),
            tol=0.000001,
        ),
        torch.tensor([0.0]),
    )

    torch.testing.assert_close(
        beignet.polynomials.multiply_physicists_hermite_polynomial_by_x(
            torch.tensor([1.0]),
        ),
        torch.tensor([0.0, 0.5]),
    )

    for i in range(1, 5):
        torch.testing.assert_close(
            beignet.polynomials.multiply_physicists_hermite_polynomial_by_x(
                torch.tensor([0.0] * i + [1.0]),
            ),
            torch.tensor([0.0] * (i - 1) + [i, 0.0, 0.5]),
        )
