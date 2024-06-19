import beignet.polynomial
import beignet.polynomial._herm2poly
import beignet.polynomial._hermfromroots
import beignet.polynomial._hermtrim
import beignet.polynomial._hermval
import numpy


def test_hermfromroots():
    res = beignet.polynomial._hermfromroots.hermfromroots([])
    numpy.testing.assert_almost_equal(
        beignet.polynomial._hermtrim.hermtrim(res, tolerance=1e-6), [1]
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial._hermfromroots.hermfromroots(roots)
        res = beignet.polynomial._hermval.hermval(roots, pol)
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._herm2poly.herm2poly(pol)[-1], 1
        )
        numpy.testing.assert_almost_equal(res, tgt)
