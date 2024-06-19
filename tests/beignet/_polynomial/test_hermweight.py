import beignet.polynomial
import numpy


def test_hermweight():
    x = numpy.linspace(-5, 5, 11)
    tgt = numpy.exp(-(x**2))
    res = beignet.polynomial.hermweight(x)
    numpy.testing.assert_almost_equal(res, tgt)
