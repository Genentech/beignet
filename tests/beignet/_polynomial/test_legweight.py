import beignet.polynomial
import beignet.polynomial._legweight
import numpy


def test_legweight():
    numpy.testing.assert_almost_equal(
        beignet.polynomial._legweight.legweight(numpy.linspace(-1, 1, 11)), 1.0
    )
