import beignet.polynomial
import numpy


def test_lagweight():
    x = numpy.linspace(0, 10, 11)
    tgt = numpy.exp(-x)
    res = beignet.polynomial.lagweight(x)
    numpy.testing.assert_almost_equal(res, tgt)
