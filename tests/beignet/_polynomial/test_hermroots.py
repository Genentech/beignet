import beignet.polynomial
import beignet.polynomial._hermfromroots
import beignet.polynomial._physicists_hermite_series_roots
import beignet.polynomial._trim_physicists_hermite_series
import numpy


def test_hermroots():
    numpy.testing.assert_almost_equal(
        beignet.polynomial._hermroots.physicists_hermite_series_roots([1]), []
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial._hermroots.physicists_hermite_series_roots([1, 1]), [-0.5]
    )
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.polynomial._hermroots.physicists_hermite_series_roots(
            beignet.polynomial._hermfromroots.hermfromroots(tgt)
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                res, tolerance=1e-6
            ),
            beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                tgt, tolerance=1e-6
            ),
        )
