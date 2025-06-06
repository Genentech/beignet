import torch

import beignet


def test_divide_probabilists_hermite_polynomial():
    for j in range(5):
        for k in range(5):
            input = torch.tensor([0.0] * j + [1.0])
            other = torch.tensor([0.0] * k + [1.0])

            quotient, remainder = beignet.divide_probabilists_hermite_polynomial(
                beignet.add_probabilists_hermite_polynomial(
                    input,
                    other,
                ),
                input,
            )

            torch.testing.assert_close(
                beignet.trim_probabilists_hermite_polynomial_coefficients(
                    beignet.add_probabilists_hermite_polynomial(
                        beignet.multiply_probabilists_hermite_polynomial(
                            quotient,
                            input,
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                beignet.trim_probabilists_hermite_polynomial_coefficients(
                    beignet.add_probabilists_hermite_polynomial(
                        input,
                        other,
                    ),
                    tol=0.000001,
                ),
            )
