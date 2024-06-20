import functools

import beignet.polynomial
import torch


def test_polypow():
    for j in range(5):
        for k in range(5):
            c = torch.arange(j + 1)
            torch.testing.assert_close(
                beignet.polynomial.polytrim(
                    beignet.polynomial.polypow(c, k),
                    tolerance=1e-6,
                ),
                beignet.polynomial.polytrim(
                    functools.reduce(
                        beignet.polynomial.polymul,
                        torch.tensor([c] * k),
                        torch.tensor([1]),
                    ),
                    tolerance=1e-6,
                ),
            )
