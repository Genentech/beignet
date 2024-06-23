import functools

import beignet.polynomial
import torch


def test_pow_physicists_hermite_series():
    for i in range(5):
        for j in range(5):
            c = torch.arange(i + 1)
            torch.testing.assert_close(
                beignet.polynomial.trim_physicists_hermite_series(
                    beignet.polynomial.pow_physicists_hermite_series(
                        c,
                        j,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_physicists_hermite_series(
                    functools.reduce(
                        beignet.polynomial.multiply_physicists_hermite_series,
                        torch.tensor([*c] * j),
                        torch.tensor([1]),
                    ),
                    tolerance=1e-6,
                ),
            )
