import beignet.polynomial
import numpy
import torch

from .test_polynomial import chebyshev_polynomial_coefficients


def test_chebyshev_series_to_power_series():
    for i in range(10):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.chebyshev_series_to_power_series(
                torch.tensor([0] * i + [1]),
            ),
            chebyshev_polynomial_coefficients[i],
        )
