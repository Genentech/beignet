import beignet.polynomial
import numpy


def test_chebweight():
    x = numpy.linspace(-1, 1, 11)[1:-1]
    numpy.testing.assert_almost_equal(
        beignet.polynomial.chebweight(x), 1.0 / (numpy.sqrt(1 + x) * numpy.sqrt(1 - x))
    )
