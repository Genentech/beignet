import beignet.polynomial
import beignet.polynomial._evaluate_laguerre_series_1d
import beignet.polynomial._lagfromroots
import beignet.polynomial._laguerre_series_to_power_series
import beignet.polynomial._trim_laguerre_series
import numpy


def test_lagfromroots():
    numpy.testing.assert_almost_equal(
        beignet.polynomial._lagtrim.trim_laguerre_series(
            beignet.polynomial._lagfromroots.lagfromroots([]), tolerance=1e-6
        ),
        [1],
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial._lagfromroots.lagfromroots(roots)
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lag2poly.laguerre_series_to_power_series(pol)[-1], 1
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagval.evaluate_laguerre_series_1d(roots, pol), 0
        )
