import functools

import beignet.polynomial
import torch


def test_polypow():
    for i in range(5):
        for j in range(5):
            c = torch.arange(i + 1)
            torch.testing.assert_close(
                beignet.polynomial.polytrim(
                    beignet.polynomial.polypow(c, j),
                    tolerance=1e-6,
                ),
                beignet.polynomial.polytrim(
                    functools.reduce(
                        beignet.polynomial.polymul,
                        torch.tensor([c] * j),
                        torch.tensor([1]),
                    ),
                    tolerance=1e-6,
                ),
            )
