import beignet
import torch


def test_multiply_legendre_polynomial():
    for i in range(5):
        input = torch.linspace(-1, 1, 100)

        a = beignet.evaluate_legendre_polynomial(
            input,
            torch.tensor([0.0] * i + [1.0]),
        )

        for j in range(5):
            b = beignet.evaluate_legendre_polynomial(
                input,
                torch.tensor([0.0] * j + [1.0]),
            )

            torch.testing.assert_close(
                beignet.evaluate_legendre_polynomial(
                    input,
                    beignet.multiply_legendre_polynomial(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
                    ),
                ),
                a * b,
            )
