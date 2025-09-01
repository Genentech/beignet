import torch

import beignet.polynomials


def test_divide_laguerre_polynomial():
    for i in range(5):
        for j in range(5):
            quotient, remainder = beignet.polynomials.divide_laguerre_polynomial(
                beignet.polynomials.add_laguerre_polynomial(
                    torch.tensor([0.0] * i + [1.0]),
                    torch.tensor([0.0] * j + [1.0]),
                ),
                torch.tensor([0.0] * i + [1.0]),
            )

            torch.testing.assert_close(
                beignet.polynomials.trim_laguerre_polynomial_coefficients(
                    beignet.polynomials.add_laguerre_polynomial(
                        beignet.polynomials.multiply_laguerre_polynomial(
                            quotient,
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                beignet.polynomials.trim_laguerre_polynomial_coefficients(
                    beignet.polynomials.add_laguerre_polynomial(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
            )
