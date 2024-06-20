import functools

import beignet.polynomial
import torch


def test_chebpow():
    for j in range(5):
        for k in range(5):
            c = torch.arange(j + 1)

            tgt = functools.reduce(
                beignet.polynomial._chebmul.multiply_chebyshev_series,
                torch.tensor([c] * k),
                torch.tensor([1]),
            )

            torch.testing.assert_close(
                beignet.polynomial.trim_chebyshev_series(
                    beignet.polynomial._chebpow.chebpow(c, k),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_chebyshev_series(
                    tgt,
                    tolerance=1e-6,
                ),
            )
