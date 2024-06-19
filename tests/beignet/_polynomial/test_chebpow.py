import functools

import beignet.polynomial
import beignet.polynomial._chebmul
import beignet.polynomial._chebpow
import beignet.polynomial._chebtrim
import torch


def test_chebpow():
    for j in range(5):
        for k in range(5):
            c = torch.arange(j + 1)

            tgt = functools.reduce(
                beignet.polynomial._chebmul.chebmul,
                torch.tensor([c] * k),
                torch.tensor([1]),
            )

            torch.testing.assert_close(
                beignet.polynomial.chebtrim(
                    beignet.polynomial._chebpow.chebpow(c, k),
                    tolerance=1e-6,
                ),
                beignet.polynomial.chebtrim(
                    tgt,
                    tolerance=1e-6,
                ),
            )
