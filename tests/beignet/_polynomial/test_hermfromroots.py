import beignet.polynomial
import beignet.polynomial._evaluate_physicists_hermite_series_1d
import beignet.polynomial._hermfromroots
import beignet.polynomial._physicists_hermite_series_to_power_series
import beignet.polynomial._trim_physicists_hermite_series
import numpy


def test_hermfromroots():
    res = beignet.polynomial._hermfromroots.hermfromroots([])
    numpy.testing.assert_almost_equal(
        beignet.polynomial._hermtrim.trim_physicists_hermite_series(
            res, tolerance=1e-6
        ),
        [1],
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial._hermfromroots.hermfromroots(roots)
        res = beignet.polynomial._hermval.evaluate_physicists_hermite_series_1d(
            roots, pol
        )
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._herm2poly.physicists_hermite_series_to_power_series(
                pol
            )[-1],
            1,
        )
        numpy.testing.assert_almost_equal(res, tgt)
