import beignet.polynomial
import beignet.polynomial._chebadd
import beignet.polynomial._chebtrim
import torch


def test_chebadd():
    for i in range(5):
        for j in range(5):
            tgt = torch.zeros(max(i, j) + 1, dtype=torch.float64)
            tgt[i] += 1
            tgt[j] += 1
            torch.testing.assert_close(
                beignet.polynomial._chebtrim.chebtrim(
                    beignet.polynomial._chebadd.chebadd(
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
