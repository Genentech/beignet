import beignet.polynomial
import torch


def test_subtract_physicists_hermite_series():
    for i in range(5):
        for j in range(5):
            tgt = torch.zeros(max(i, j) + 1, dtype=torch.float64)
            tgt[i] += 1
            tgt[j] -= 1
            torch.testing.assert_close(
                beignet.polynomial.trim_physicists_hermite_series(
                    beignet.polynomial.subtract_physicists_hermite_series(
                        torch.tensor([0] * i + [1]),
                        torch.tensor([0] * j + [1]),
                    ),
                    tolerance=0.000001,
                ),
                beignet.polynomial.trim_physicists_hermite_series(
                    tgt,
                    tolerance=0.000001,
                ),
            )
