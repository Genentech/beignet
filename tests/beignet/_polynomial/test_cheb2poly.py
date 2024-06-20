import beignet.polynomial
import beignet.polynomial._chebyshev_series_to_power_series
import numpy
import torch

from .test_polynomial import chebyshev_polynomial_coefficients


def test_cheb2poly():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial._cheb2poly.chebyshev_series_to_power_series(
                torch.tensor([0] * i + [1]),
            ),
            chebyshev_polynomial_coefficients[i],
        )
