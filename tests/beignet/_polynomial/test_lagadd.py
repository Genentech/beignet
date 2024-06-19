import beignet.polynomial
import beignet.polynomial._lagadd
import beignet.polynomial._lagtrim
import torch


def test_lagadd():
    for i in range(5):
        for j in range(5):
            tgt = torch.zeros(max(i, j) + 1, dtype=torch.float64)
            tgt[i] += 1
            tgt[j] += 1
            torch.testing.assert_close(
                beignet.polynomial._lagtrim.lagtrim(
                    beignet.polynomial._lagadd.lagadd(
                        torch.tensor([0] * i + [1]),
                        torch.tensor([0] * j + [1]),
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial._lagtrim.lagtrim(
                    tgt,
                    tolerance=1e-6,
                ),
            )
