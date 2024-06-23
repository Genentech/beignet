import beignet.polynomial
import beignet.polynomial._legendre_series_weight
import numpy


def test_legweight():
    numpy.testing.assert_almost_equal(
        beignet.polynomial._legweight.legendre_series_weight(numpy.linspace(-1, 1, 11)),
        1.0,
    )
