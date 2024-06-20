import functools

import beignet.polynomial
import beignet.polynomial._polymul
import beignet.polynomial._polypow
import beignet.polynomial._polytrim
import torch


def test_polypow():
    for i in range(5):
        for j in range(5):
            c = torch.arange(i + 1)
            torch.testing.assert_close(
                beignet.polynomial._polytrim.polytrim(
                    beignet.polynomial._polypow.polypow(c, j),
                    tolerance=1e-6,
                ),
                beignet.polynomial._polytrim.polytrim(
                    functools.reduce(
                        beignet.polynomial.polymul,
                        torch.tensor([c] * j),
                        torch.tensor([1]),
                    ),
                    tolerance=1e-6,
                ),
            )
