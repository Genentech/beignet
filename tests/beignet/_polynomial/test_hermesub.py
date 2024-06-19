import beignet.polynomial
import beignet.polynomial._hermesub
import torch


def test_hermesub():
    for i in range(5):
        for j in range(5):
            tgt = torch.zeros(max(i, j) + 1, dtype=torch.float64)
            tgt[i] += 1
            tgt[j] -= 1
            torch.testing.assert_close(
                beignet.polynomial.hermetrim(
                    beignet.polynomial._hermesub.hermesub(
                        torch.tensor([0] * i + [1]),
                        torch.tensor([0] * j + [1]),
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.hermetrim(
                    tgt,
                    tolerance=1e-6,
                ),
            )
