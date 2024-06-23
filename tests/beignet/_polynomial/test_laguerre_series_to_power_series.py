import beignet.polynomial
import beignet.polynomial._laguerre_series_to_power_series
import numpy

from .test_polynomial import laguerre_polynomial_coefficients


def test_laguerre_series_to_power_series():
    for i in range(7):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.laguerre_series_to_power_series([0] * i + [1]),
            laguerre_polynomial_coefficients[i],
        )
