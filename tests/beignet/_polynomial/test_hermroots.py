import beignet.polynomial
import beignet.polynomial._hermfromroots
import beignet.polynomial._hermroots
import beignet.polynomial._hermtrim
import numpy


def test_hermroots():
    numpy.testing.assert_almost_equal(beignet.polynomial._hermroots.hermroots([1]), [])
    numpy.testing.assert_almost_equal(
        beignet.polynomial._hermroots.hermroots([1, 1]), [-0.5]
    )
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.polynomial._hermroots.hermroots(
            beignet.polynomial._hermfromroots.hermfromroots(tgt)
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermtrim.hermtrim(res, tolerance=1e-6),
            beignet.polynomial._hermtrim.hermtrim(tgt, tolerance=1e-6),
        )
