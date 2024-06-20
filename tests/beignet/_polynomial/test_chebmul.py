import beignet.polynomial
import beignet.polynomial._multiply_chebyshev_series
import beignet.polynomial._trim_chebyshev_series
import torch


def test_multiply_chebyshev_series():
    for j in range(5):
        for k in range(5):
            output = torch.zeros(j + k + 1, dtype=torch.float64)

            output[j + k] += 0.5
            output[abs(j - k)] += 0.5

            torch.testing.assert_close(
                beignet.polynomial.trim_chebyshev_series(
                    beignet.polynomial.multiply_chebyshev_series(
                        torch.tensor([0] * j + [1]),
                        torch.tensor([0] * k + [1]),
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_chebyshev_series(
                    output,
                    tolerance=1e-6,
                ),
            )
