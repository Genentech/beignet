import functools

import beignet.polynomial
import torch


def test_pow_physicists_hermite_series():
    for j in range(5):
        for k in range(5):
            torch.testing.assert_close(
                beignet.polynomial.trim_physicists_hermite_series(
                    beignet.polynomial.pow_physicists_hermite_series(
                        torch.arange(j + 1),
                        k,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_physicists_hermite_series(
                    functools.reduce(
                        beignet.polynomial.multiply_physicists_hermite_series,
                        torch.tensor([*torch.arange(j + 1)] * k),
                        torch.tensor([1]),
                    ),
                    tolerance=1e-6,
                ),
            )
