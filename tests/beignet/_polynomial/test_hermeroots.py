import beignet.polynomial
import beignet.polynomial._hermefromroots
import beignet.polynomial._hermeroots
import beignet.polynomial._hermetrim
import numpy


def test_hermeroots():
    numpy.testing.assert_almost_equal(
        beignet.polynomial._hermeroots.hermeroots([1]), []
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial._hermeroots.hermeroots([1, 1]), [-1]
    )
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.polynomial._hermeroots.hermeroots(
            beignet.polynomial._hermefromroots.hermefromroots(tgt)
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._hermetrim.hermetrim(res, tolerance=1e-6),
            beignet.polynomial._hermetrim.hermetrim(tgt, tolerance=1e-6),
        )
