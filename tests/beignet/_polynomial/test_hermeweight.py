import beignet.polynomial
import beignet.polynomial._hermeweight
import numpy


def test_hermeweight():
    x = numpy.linspace(-5, 5, 11)
    tgt = numpy.exp(-0.5 * x**2)
    res = beignet.polynomial._hermeweight.hermeweight(x)
    numpy.testing.assert_almost_equal(res, tgt)
