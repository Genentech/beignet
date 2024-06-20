import beignet.polynomial
import beignet.polynomial._lagfromroots
import beignet.polynomial._lagroots
import beignet.polynomial._trim_laguerre_series
import numpy


def test_lagroots():
    numpy.testing.assert_almost_equal(beignet.polynomial._lagroots.lagroots([1]), [])
    numpy.testing.assert_almost_equal(
        beignet.polynomial._lagroots.lagroots([0, 1]), [1]
    )
    for i in range(2, 5):
        tgt = numpy.linspace(0, 3, i)
        res = beignet.polynomial._lagroots.lagroots(
            beignet.polynomial._lagfromroots.lagfromroots(tgt)
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagtrim.trim_laguerre_series(res, tolerance=1e-6),
            beignet.polynomial._lagtrim.trim_laguerre_series(tgt, tolerance=1e-6),
        )
