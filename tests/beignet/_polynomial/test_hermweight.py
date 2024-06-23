import beignet.polynomial
import beignet.polynomial._physicists_hermite_series_weight
import numpy


def test_hermweight():
    x = numpy.linspace(-5, 5, 11)
    tgt = numpy.exp(-(x**2))
    res = beignet.polynomial._hermweight.physicists_hermite_series_weight(x)
    numpy.testing.assert_almost_equal(res, tgt)
