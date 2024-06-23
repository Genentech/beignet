import functools

import beignet.polynomial
import torch


def test_polypow():
    for j in range(5):
        for k in range(5):
            c = torch.arange(j + 1)
            torch.testing.assert_close(
                beignet.polynomial.trim_power_series(
                    beignet.polynomial.pow_power_series(c, k),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_power_series(
                    functools.reduce(
                        beignet.polynomial.multiply_power_series,
                        torch.tensor([c] * k),
                        torch.tensor([1]),
                    ),
                    tolerance=1e-6,
                ),
            )
