import torch

import beignet.polynomials


def test_legendre_polynomial_roots():
    torch.testing.assert_close(
        beignet.polynomials.legendre_polynomial_roots(
            torch.tensor([1.0]),
        ),
        torch.tensor([]),
    )

    torch.testing.assert_close(
        beignet.polynomials.legendre_polynomial_roots(torch.tensor([1.0, 2.0])),
        torch.tensor([-0.5]),
    )

    for index in range(2, 5):
        torch.testing.assert_close(
            beignet.polynomials.trim_legendre_polynomial_coefficients(
                beignet.polynomials.legendre_polynomial_roots(
                    beignet.polynomials.legendre_polynomial_from_roots(
                        torch.linspace(-1, 1, index),
                    ),
                ),
                tol=0.000001,
            ),
            beignet.polynomials.trim_legendre_polynomial_coefficients(
                torch.linspace(-1, 1, index),
                tol=0.000001,
            ),
        )
