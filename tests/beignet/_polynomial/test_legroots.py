import beignet.polynomial
import beignet.polynomial._legendre_series_roots
import beignet.polynomial._legfromroots
import beignet.polynomial._trim_legendre_series
import numpy


def test_legroots():
    numpy.testing.assert_almost_equal(
        beignet.polynomial._legroots.legendre_series_roots([1]), []
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial._legroots.legendre_series_roots([1, 2]), [-0.5]
    )
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.polynomial._legroots.legendre_series_roots(
            beignet.polynomial._legfromroots.legfromroots(tgt)
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legtrim.trim_legendre_series(res, tolerance=1e-6),
            beignet.polynomial._legtrim.trim_legendre_series(tgt, tolerance=1e-6),
        )
