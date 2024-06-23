import beignet.polynomial
import torch


def test_multiply_power_series():
    for j in range(5):
        for k in range(5):
            output = torch.zeros(j + k + 1, dtype=torch.float64)

            output[j + k] = output[j + k] + 1

            torch.testing.assert_close(
                beignet.polynomial.trim_power_series(
                    beignet.polynomial.multiply_power_series(
                        torch.tensor([0] * j + [1]),
                        torch.tensor([0] * k + [1]),
                    ),
                    tolerance=0.000001,
                ),
                beignet.polynomial.trim_power_series(
                    output,
                    tolerance=0.000001,
                ),
            )
