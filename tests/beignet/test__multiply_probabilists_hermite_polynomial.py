import beignet
import torch


def test_multiply_probabilists_hermite_polynomial():
    for index in range(5):
        input = torch.linspace(-3, 3, 100)

        val1 = beignet.evaluate_probabilists_hermite_polynomial(
            input,
            torch.tensor([0.0] * index + [1.0]),
        )

        for k in range(5):
            val2 = beignet.evaluate_probabilists_hermite_polynomial(
                input,
                torch.tensor([0.0] * k + [1.0]),
            )

            torch.testing.assert_close(
                beignet.evaluate_probabilists_hermite_polynomial(
                    input,
                    beignet.multiply_probabilists_hermite_polynomial(
                        torch.tensor([0.0] * index + [1.0]),
                        torch.tensor([0.0] * k + [1.0]),
                    ),
                ),
                val1 * val2,
            )
