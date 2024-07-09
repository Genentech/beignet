import functools

import beignet
import torch


def test_laguerre_polynomial_power():
    for i in range(5):
        for j in range(5):
            torch.testing.assert_close(
                beignet.trim_laguerre_polynomial_coefficients(
                    beignet.laguerre_polynomial_power(
                        torch.arange(0.0, i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                beignet.trim_laguerre_polynomial_coefficients(
                    functools.reduce(
                        beignet.multiply_laguerre_polynomial,
                        [torch.arange(0.0, i + 1)] * j,
                        torch.tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )
