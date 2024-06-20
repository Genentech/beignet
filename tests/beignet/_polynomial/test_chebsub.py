import beignet.polynomial
import beignet.polynomial._chebtrim
import beignet.polynomial._subtract_chebyshev_series
import torch


def test_chebsub():
    for i in range(5):
        for j in range(5):
            tgt = torch.zeros(max(i, j) + 1, dtype=torch.float64)
            tgt[i] += 1
            tgt[j] -= 1
            torch.testing.assert_close(
                beignet.polynomial._chebtrim.chebtrim(
                    beignet.polynomial._chebsub.subtract_chebyshev_series(
                        torch.tensor([0] * i + [1]),
                        torch.tensor([0] * j + [1]),
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial._chebtrim.chebtrim(
                    tgt,
                    tolerance=1e-6,
                ),
            )
