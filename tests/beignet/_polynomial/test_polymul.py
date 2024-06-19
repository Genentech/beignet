import beignet.polynomial
import torch


def test_polymul():
    for j in range(5):
        for k in range(5):
            output = torch.zeros(j + k + 1, dtype=torch.float64)

            output[j + k] = output[j + k] + 1

            torch.testing.assert_close(
                beignet.polynomial.polytrim(
                    beignet.polynomial.polymul(
                        torch.tensor([0] * j + [1]),
                        torch.tensor([0] * k + [1]),
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.polytrim(
                    output,
                    tolerance=1e-6,
                ),
            )
