import beignet.polynomial
import numpy


def test_legweight():
    numpy.testing.assert_almost_equal(
        beignet.polynomial.legweight(numpy.linspace(-1, 1, 11)), 1.0
    )
