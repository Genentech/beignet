import beignet.polynomial
import beignet.polynomial._subtract_laguerre_series
import beignet.polynomial._trim_laguerre_series
import torch


def test_lagsub():
    for i in range(5):
        for j in range(5):
            tgt = torch.zeros(max(i, j) + 1, dtype=torch.float64)
            tgt[i] += 1
            tgt[j] -= 1
            torch.testing.assert_close(
                beignet.polynomial._lagtrim.trim_laguerre_series(
                    beignet.polynomial._lagsub.subtract_laguerre_series(
                        torch.tensor([0] * i + [1]),
                        torch.tensor([0] * j + [1]),
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial._lagtrim.trim_laguerre_series(
                    tgt,
                    tolerance=1e-6,
                ),
            )
