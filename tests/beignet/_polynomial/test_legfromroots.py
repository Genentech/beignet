import beignet.polynomial
import beignet.polynomial._evaluate_legendre_series_1d
import beignet.polynomial._legendre_series_to_power_series
import beignet.polynomial._legfromroots
import beignet.polynomial._trim_legendre_series
import numpy


def test_legfromroots():
    res = beignet.polynomial._legfromroots.legfromroots([])
    numpy.testing.assert_almost_equal(
        beignet.polynomial._legtrim.trim_legendre_series(res, tolerance=1e-6), [1]
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial._legfromroots.legfromroots(roots)
        res = beignet.polynomial._legval.evaluate_legendre_series_1d(roots, pol)
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._leg2poly.legendre_series_to_power_series(pol)[-1], 1
        )
        numpy.testing.assert_almost_equal(res, tgt)
