import functools

import torch

import beignet.polynomials


def test_probabilists_hermite_polynomial_power():
    for j in range(5):
        for k in range(5):
            torch.testing.assert_close(
                beignet.polynomials.trim_probabilists_hermite_polynomial_coefficients(
                    beignet.polynomials.probabilists_hermite_polynomial_power(
                        torch.arange(0.0, j + 1),
                        k,
                    ),
                    tol=0.000001,
                ),
                beignet.polynomials.trim_probabilists_hermite_polynomial_coefficients(
                    functools.reduce(
                        beignet.polynomials.multiply_probabilists_hermite_polynomial,
                        [torch.arange(0.0, j + 1)] * k,
                        torch.tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )
