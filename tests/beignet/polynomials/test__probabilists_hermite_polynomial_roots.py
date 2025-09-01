import torch

import beignet.polynomials


def test_probabilists_hermite_polynomial_roots():
    torch.testing.assert_close(
        beignet.polynomials.probabilists_hermite_polynomial_roots(
            torch.tensor([1.0]),
        ),
        torch.tensor([]),
    )

    torch.testing.assert_close(
        beignet.polynomials.probabilists_hermite_polynomial_roots(
            torch.tensor([1.0, 1.0]),
        ),
        torch.tensor([-1.0]),
    )

    for index in range(2, 5):
        input = torch.linspace(-1, 1, index)

        torch.testing.assert_close(
            beignet.polynomials.trim_probabilists_hermite_polynomial_coefficients(
                beignet.polynomials.probabilists_hermite_polynomial_roots(
                    beignet.polynomials.probabilists_hermite_polynomial_from_roots(
                        input,
                    )
                ),
                tol=0.000001,
            ),
            beignet.polynomials.trim_probabilists_hermite_polynomial_coefficients(
                input,
                tol=0.000001,
            ),
        )
