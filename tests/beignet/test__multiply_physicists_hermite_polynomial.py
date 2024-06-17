import beignet
import torch


def test_multiply_physicists_hermite_polynomial():
    for i in range(5):
        input = torch.linspace(-3, 3, 100)

        val1 = beignet.evaluate_physicists_hermite_polynomial(
            input,
            torch.tensor([0.0] * i + [1.0]),
        )

        for j in range(5):
            val2 = beignet.evaluate_physicists_hermite_polynomial(
                input,
                torch.tensor([0.0] * j + [1.0]),
            )

            torch.testing.assert_close(
                beignet.evaluate_physicists_hermite_polynomial(
                    input,
                    beignet.multiply_physicists_hermite_polynomial(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
                    ),
                ),
                val1 * val2,
            )
