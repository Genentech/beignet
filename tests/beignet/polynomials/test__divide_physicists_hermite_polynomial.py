import torch

import beignet.polynomials


def test_divide_physicists_hermite_polynomial():
    for j in range(5):
        for k in range(5):
            input = torch.tensor([0.0] * j + [1.0])
            other = torch.tensor([0.0] * k + [1.0])

            quotient, remainder = (
                beignet.polynomials.divide_physicists_hermite_polynomial(
                    beignet.polynomials.add_physicists_hermite_polynomial(
                        input,
                        other,
                    ),
                    input,
                )
            )

            torch.testing.assert_close(
                beignet.polynomials.trim_physicists_hermite_polynomial_coefficients(
                    beignet.polynomials.add_physicists_hermite_polynomial(
                        beignet.polynomials.multiply_physicists_hermite_polynomial(
                            quotient,
                            input,
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                beignet.polynomials.trim_physicists_hermite_polynomial_coefficients(
                    beignet.polynomials.add_physicists_hermite_polynomial(
                        input,
                        other,
                    ),
                    tol=0.000001,
                ),
            )
