import beignet.polynomial
import beignet.polynomial._evaluate_probabilists_hermite_series_1d
import beignet.polynomial._hermefromroots
import beignet.polynomial._probabilists_hermite_series_to_power_series
import beignet.polynomial._trim_probabilists_hermite_series
import numpy


def test_hermefromroots():
    res = beignet.polynomial._hermefromroots.hermefromroots([])
    numpy.testing.assert_almost_equal(
        beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
            res, tolerance=1e-6
        ),
        [1],
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial._hermefromroots.hermefromroots(roots)
        res = beignet.polynomial._hermeval.evaluate_probabilists_hermite_series_1d(
            roots, pol
        )
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._herme2poly.probabilists_hermite_series_to_power_series(
                pol
            )[-1],
            1,
        )
        numpy.testing.assert_almost_equal(res, tgt)
