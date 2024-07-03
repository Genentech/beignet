import beignet
import torch

torch.set_default_dtype(torch.float64)


def test_multiply_laguerre_polynomial():
    for i in range(5):
        input = torch.linspace(-3, 3, 100)

        a = beignet.evaluate_laguerre_polynomial(
            input,
            torch.tensor([0.0] * i + [1.0]),
        )

        for j in range(5):
            b = beignet.evaluate_laguerre_polynomial(
                input,
                torch.tensor([0.0] * j + [1.0]),
            )

            torch.testing.assert_close(
                beignet.evaluate_laguerre_polynomial(
                    input,
                    beignet.trim_laguerre_polynomial_coefficients(
                        beignet.multiply_laguerre_polynomial(
                            torch.tensor([0.0] * i + [1.0]),
                            torch.tensor([0.0] * j + [1.0]),
                        ),
                    ),
                ),
                a * b,
            )


def test_lagline():
    torch.testing.assert_close(
        beignet.linear_laguerre_polynomial(3.0, 4.0),
        torch.tensor([7.0, -4.0]),
    )
