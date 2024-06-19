import beignet.polynomial
import beignet.polynomial._hermsub
import beignet.polynomial._hermtrim
import torch


def test_hermsub():
    for i in range(5):
        for j in range(5):
            tgt = torch.zeros(max(i, j) + 1, dtype=torch.float64)
            tgt[i] += 1
            tgt[j] -= 1
            torch.testing.assert_close(
                beignet.polynomial._hermtrim.hermtrim(
                    beignet.polynomial._hermsub.hermsub(
                        torch.tensor([0] * i + [1]),
                        torch.tensor([0] * j + [1]),
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial._hermtrim.hermtrim(
                    tgt,
                    tolerance=1e-6,
                ),
            )
