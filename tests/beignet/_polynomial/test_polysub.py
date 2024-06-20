import beignet.polynomial
import torch


def test_polysub():
    for j in range(5):
        for k in range(5):
            output = torch.zeros(max(j, k) + 1, dtype=torch.float64)

            output[j] += 1
            output[k] -= 1

            torch.testing.assert_close(
                beignet.polynomial.trim_power_series(
                    beignet.polynomial.subtract_power_series(
                        torch.tensor([0] * j + [1]),
                        torch.tensor([0] * k + [1]),
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_power_series(
                    output,
                    tolerance=1e-6,
                ),
            )
