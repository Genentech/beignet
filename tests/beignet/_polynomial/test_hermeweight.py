import beignet.polynomial
import numpy


def test_hermeweight():
    x = numpy.linspace(-5, 5, 11)
    tgt = numpy.exp(-0.5 * x**2)
    res = beignet.polynomial.probabilists_hermite_series_weight(x)
    numpy.testing.assert_almost_equal(res, tgt)
