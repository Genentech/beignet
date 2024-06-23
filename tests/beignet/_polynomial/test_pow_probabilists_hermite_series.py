import functools

import beignet.polynomial
import torch


def test_pow_probabilists_hermite_series():
    for j in range(5):
        for k in range(5):
            c = torch.arange(j + 1)
            torch.testing.assert_close(
                beignet.polynomial.trim_probabilists_hermite_series(
                    beignet.polynomial.pow_probabilists_hermite_series(c, k),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_probabilists_hermite_series(
                    functools.reduce(
                        beignet.polynomial.multiply_probabilists_hermite_series,
                        [c] * k,
                        torch.tensor([1]),
                    ),
                    tolerance=1e-6,
                ),
            )
