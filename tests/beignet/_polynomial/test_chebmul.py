import beignet.polynomial
import beignet.polynomial._chebtrim
import torch


def test_chebmul():
    for j in range(5):
        for k in range(5):
            output = torch.zeros(j + k + 1, dtype=torch.float64)

            output[j + k] += 0.5
            output[abs(j - k)] += 0.5

            torch.testing.assert_close(
                beignet.polynomial.chebtrim(
                    beignet.polynomial.chebmul(
                        torch.tensor([0] * j + [1]),
                        torch.tensor([0] * k + [1]),
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.chebtrim(
                    output,
                    tolerance=1e-6,
                ),
            )
